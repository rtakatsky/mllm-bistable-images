import glob
import os

from utils import PROJECT_ROOT as base_dir  # utils sets HF_HOME before transformers is imported

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import (
    AutoProcessor,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Blip2ForConditionalGeneration,
    Blip2Processor
    )
from transformers import Blip2Processor
from typing import List, Dict, Optional, Union, Literal
import random
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib as mpl
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def model_resampling_ablation(model, processor, source_image, target_image, source_prompt, target_prompt, tokens_of_interest, device="cpu", save_dir=None, ablation_keys=["resid_post", "attn_out", "mlp_out"], text_only=False):
    # assume source_prompt and target_prompt are the same length
    # text_only: patch only non-image token positions (~20x cheaper; the paper's
    # top_down_patching_every figures use text-token panels only). Skipped
    # positions are NaN in the store, and caches are kept separate (see below).

    #         print(f"Loading cache from {save_dir}")
    #         return cache

    model = model.to(device)
    model.eval()

    llm = model.language_model.model
    results = {}

    # preprocess
    source_inputs = processor(text=source_prompt, images=source_image, return_tensors="pt").to(device)
    target_inputs = processor(text=target_prompt, images=target_image, return_tensors="pt").to(device)

    # require same sequence length so token-wise swap is well-defined
    S_src = source_inputs["input_ids"].shape[1]
    S_tgt = target_inputs["input_ids"].shape[1]
    assert S_src == S_tgt, "Source and target sequence lengths must match for resampling ablation"
    print(f"Source sequence length: {S_src}, Target sequence length: {S_tgt}")

    if text_only:
        img_tok_id = model.config.image_token_index
        ids = target_inputs["input_ids"][0]
        positions = [t for t in range(S_src) if int(ids[t]) != img_tok_id]
        assert 0 < len(positions) < S_src, "text_only found no image tokens to skip (unexpanded input_ids?)"
        print(f"text_only: patching {len(positions)}/{S_src} positions")
    else:
        positions = list(range(S_src))

    interesting_ids = torch.tensor(list(tokens_of_interest.values()), device=device)

    def run_forward(inputs):
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]  # after the final norm
            logits = model.language_model.lm_head(last_hidden)  # (B,S,V)
            return logits[0, -1, interesting_ids].detach().cpu().numpy()  # (T)
    
    results["original"] = run_forward(target_inputs)
    # Only these keys are consumed by the resampling hooks below; the previous
    # default (all ten keys, incl. the ~10 GB head_out) was also moved wholesale
    # onto the GPU in the loop below, OOMing 40-layer models.
    source_cache = get_cache(model, processor, source_image, source_prompt, device, save_dir,
                             cache_keys=["embed", "resid_post", "attn_out", "mlp_out"])
    for key in source_cache.keys():
        if key == "embed":
            print(f"embed.shape: {source_cache[key].shape}")
        source_cache[key] = source_cache[key].to(device)

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"original.npy"), results["original"])

    L = len(llm.layers)
    S = S_src
    T = len(interesting_ids)
    H = llm.config.num_attention_heads  # per-model (llava-1.5-13b has 40)
    d_head = llm.layers[0].self_attn.o_proj.weight.shape[0] // H

    def resample_token_out(out, t_idx, cached):
        # out: Tensor or tuple
        if isinstance(out, tuple):
            y = out[0].clone()
            y[:, t_idx, :] = cached # (d,)
            return (y,) + out[1:]
        else:
            y = out.clone()
            y[:, t_idx, :] = cached
            return y
    def resample_token_inp(inp, t_idx, cached):
        # inp: tuple
        x = inp[0].clone()
        x[:, t_idx, :] = cached
        return (x,) + inp[1:]

    def resample_head_in_o_proj(x, h_idx, cached):
        # x: (B,S,H*D)
        B, Slen, C = x.shape
        assert C == H * d_head, f"mismatch: C={C}, H*D={H*d_head}"
        if isinstance(x, tuple):
            x_new = x[0].clone()
        else:
            x_new = x.clone()
        x_view = x_new.view(B, Slen, H, d_head)  # (B,S,H,D)
        x_view[:, :, h_idx, :] = cached
        return (x_view.view(B, Slen, C),) if isinstance(x, tuple) else x_view.view(B, Slen, C)
    
    for ablation_key in ablation_keys:
        print(f"Ablating {ablation_key}")
        headwise = ablation_key == "head_out"
        # Full-position caches use the plain name; text_only caches get their
        # own suffix so a NaN-padded store is never mistaken for a
        # full one. A full cache, if present, always satisfies a text_only run.
        cache_name = f"{ablation_key}__textonly.npy" if text_only else f"{ablation_key}.npy"
        if os.path.exists(os.path.join(save_dir, f"{ablation_key}.npy")):
            print(f"Loading {ablation_key}.npy")
            ablated_store = np.load(os.path.join(save_dir, f"{ablation_key}.npy"))
            results[ablation_key] = ablated_store
            continue
        if text_only and os.path.exists(os.path.join(save_dir, cache_name)):
            print(f"Loading {cache_name}")
            ablated_store = np.load(os.path.join(save_dir, cache_name))
            results[ablation_key] = ablated_store
            continue
        make_store = (lambda shape: np.full(shape, np.nan, dtype=np.float32)) if text_only \
            else (lambda shape: np.empty(shape, dtype=np.float32))
        if not headwise:
            if ablation_key == "resid_post":
                ablated_store = make_store((L+1, S, T))
                # resid_pre at first layer
                for t in positions:
                    handle = llm.layers[0].register_forward_pre_hook(lambda m, inp, t=t: resample_token_inp(inp, t, source_cache["embed"][t, :]))
                    ablated_store[0, t, :] = run_forward(target_inputs)
                    handle.remove()
            else:
                ablated_store = make_store((L, S, T))

            for l in tqdm(range(L)):
                blk = llm.layers[l]
                for t in positions:
  # register the hook
                    if ablation_key == "resid_post":
                        handle = blk.register_forward_hook(lambda m, inp, out, t=t: resample_token_out(out, t, source_cache["resid_post"][l+1, t, :]))
                    #         lambda m, inp: (inp[0].clone().index_fill(1, torch.tensor([t], device=inp[0].device), 0),)
                    #     )
                    elif ablation_key == "attn_out":
                        handle = blk.self_attn.register_forward_hook(lambda m, inp, out, t=t: resample_token_out(out, t, source_cache["attn_out"][l, t, :]))
                    elif ablation_key == "mlp_out":
                        handle = blk.mlp.register_forward_hook(lambda m, inp, out, t=t: resample_token_out(out, t, source_cache["mlp_out"][l, t, :]))
                    else:
                        raise ValueError(f"Unknown key {ablation_key}")

                    # forward
                    ablated_logits = run_forward(target_inputs)
                    if ablation_key == "resid_post":
                        ablated_store[l+1, t, :] = ablated_logits
                    else:
                        ablated_store[l, t, :] = ablated_logits

  # remove the hook
                    handle.remove()
        else:
            ablated_store = np.empty((L, H, T), dtype=np.float32)
            for l in tqdm(range(L)):
                blk = llm.layers[l]
                for h in range(H):
                    handle = blk.self_attn.o_proj.register_forward_hook(lambda m, inp, out, h=h: resample_head_in_o_proj(inp[0], h, source_cache["head_out"][l, h, :]))
                    ablated_store[l, h, :] = run_forward(target_inputs)
                    handle.remove()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, cache_name), ablated_store)
        results[ablation_key] = ablated_store

    return results


def logitlens(
    model,
    hidden_states,
    tokens_of_interest,
    pos, # list
    softmax=False,
    head_out=False,
    device="cpu",
    save_path=None,
):
    #     return logits_of_interest

    model = model.to(device)
    model.eval()

    pos = torch.tensor(pos).to(device)
    interesting_token_ids = torch.tensor(list(tokens_of_interest.values())).to(device)

    norm = model.language_model.model.norm
    lm_head = model.language_model.lm_head

    if not head_out:
        num_layers = len(hidden_states)
        logits_of_interest = torch.zeros((num_layers, len(pos), len(tokens_of_interest))) # [L, P, T]
        for layer in range(num_layers):
            layer_hidden_states = hidden_states[layer].to(device) # [P, d]
            normalized = norm(layer_hidden_states)
            logits = lm_head(normalized) # [P, V]
            if softmax:
                logits = torch.softmax(logits, dim=-1)

            
            logits_of_interest[layer, :, :] = logits.index_select(0, pos).index_select(1, interesting_token_ids)
            #     logits_of_interest[layer, :, i] = logits[pos, idx]
    else:
        num_layers = len(hidden_states)
        num_heads = hidden_states[0].shape[1]
        logits_of_interest = torch.zeros((num_layers, len(pos), num_heads, len(tokens_of_interest)))
        for layer in range(num_layers):
            for head in range(num_heads):
                logits = lm_head(hidden_states[layer][:, head, :]).to(device)
                if softmax:
                    logits = torch.softmax(logits, dim=-1)
                logits_of_interest[layer, :, head, :] = logits.index_select(0, pos).index_select(1, token_ids)
                #     logits_of_interest[layer, :, head, i] = logits.index_select(0, pos).index_select(1, idx)
    
    logits_of_interest = logits_of_interest.detach().cpu().numpy()
    if save_path is not None:
        np.save(save_path, logits_of_interest)
    return logits_of_interest

def plot_layer_trajectory(
    logits_of_interest: np.ndarray,
    tokens_of_interest: dict[str, int],
    x_token: str,
    y_token: str,
    cmap: str = "viridis",
    annotate_layers: bool = True,
    image: Image.Image = None,
    save_path: str = None,
    title: str = None,
    softmax: bool = False,
):
    """
    logits_of_interest: shape (num_layers, num_tokens)
    tokens: list of token strings in the same order as columns of logits_of_interest
    x_token, y_token: which two tokens to use for the x/y axes
    """
    if x_token not in tokens_of_interest or y_token not in tokens_of_interest:
        raise ValueError("Requested tokens must be in the provided tokens list.")

    ix = list(tokens_of_interest.keys()).index(x_token)
    iy = list(tokens_of_interest.keys()).index(y_token)
    x = logits_of_interest[:, ix]
    y = logits_of_interest[:, iy]
    layers = np.arange(len(x))  # 0..num_layers-1

    # Build segments for gradient line
    points = np.stack([x, y], axis=1)  # (L, 2)
    segments = np.stack([points[:-1], points[1:]], axis=1)  # (L-1, 2, 2)

    norm = Normalize(vmin=0, vmax=len(layers) - 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(layers[:-1])  # color between layer i and i+1
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lc)

    # Scatter the actual layer points with same colormap
    sc = ax.scatter(x, y, c=layers, cmap=cmap, norm=norm, s=40, edgecolor="k", linewidth=0.5)

    if annotate_layers:
        for layer_idx in layers:
            ax.annotate(
                str(layer_idx),
                (x[layer_idx], y[layer_idx]),
                fontsize=6,
                xytext=(2, 2),
                textcoords="offset points",
            )
    lims = [
        min(np.nanmin(x), np.nanmin(y)),
        max(np.nanmax(x), np.nanmax(y)),
    ]
    ax.plot(lims, lims, linestyle='--', linewidth=1, color='gray')

    ax.set_xlabel(f"logit of {x_token}") if not softmax else ax.set_xlabel(f"prob of {x_token}")
    ax.set_ylabel(f"logit of {y_token}") if not softmax else ax.set_ylabel(f"prob of {y_token}")
    if title is not None:
        ax.set_title(title, loc="left")
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(sc, ax=ax, label="layer", shrink=0.5)

    if image is not None:
        # Inset image
        image_bbox = [0.39, 0.95, 0.2, 0.2]
        inset_ax = fig.add_axes(image_bbox, anchor='NE')
        inset_ax.imshow(image)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)

    plt.show()

def show_ablation_of_image_tokens(
    tokenizer,
    # prompt: str,
    image: Image.Image,
    ablated_store: np.ndarray,
    original_store: np.ndarray,
    tokens_of_interest: dict[str, int],
    show_diff: list[int] = [[0, 1]],
    cache_key: str = "resid_post",
    save_dir: str = None,
    title: str = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    ):
    
    assert cache_key in ["resid_post", "attn_out", "mlp_out", "resid_mid", "resid"], f"Invalid cache key: {cache_key}"
    if cache_key == "head_out":
        raise NotImplementedError("head_out is not implemented yet")
    
    # # Tokenize the prompt

    # Find the image token and replace it with image tokens
    img_token_count = (image_size // patch_size) ** 2  # 576 for 336x336 image with 14x14 patches

    #         # One indexed because the HTML logic wants it that way
    #         token_labels.extend([f"<IMG{(i+1):03d}>" for i in range(img_token_count)])
    #         token_labels.append(tokenizer.decode([token_id]))


    logitdiff_image = ablated_store[:, 5:5+img_token_count, :] - original_store[None, None, :]
    for i, token in enumerate(tokens_of_interest.keys()):
        # logitlens for image tokens
        fig, ax = plt.subplots(6, 6, figsize=(10, 12))

        # flatten ax
        ax = ax.flatten()

        abs_max = np.max(np.abs(logitdiff_image[:, :, i]))
        for l in range(len(ax)):
            if l >= logitdiff_image.shape[0]:
                ax[l].axis("off")
            else:
                ax[l].imshow(np.asarray(image.convert("RGB").resize((image_size, image_size))))
                logitdiff_map = logitdiff_image[l, :, i].reshape(image_size//patch_size, image_size//patch_size)
                logitdiff_map_up = np.kron(logitdiff_map, np.ones((patch_size, patch_size)))
                last_im = ax[l].imshow(logitdiff_map_up, cmap="RdBu", vmin=-abs_max, vmax=abs_max, alpha=0.8)
                ax[l].set_xticks([])
                ax[l].set_yticks([])
                if cache_key == "resid_post":
                    if l == 0:
                        ax[l].set_title(f"L{l+1}: resid_pre")
                    else:
                        ax[l].set_title(f"L{l}")
                else:
                    ax[l].set_title(f"L{l+1}")

        cbar = fig.colorbar(
            last_im, ax=ax, location="right",
            fraction=0.03,
            pad=0.02,
            aspect=30,
            shrink=0.9,
            label=f"logit diff of {token}"
        )
        if title is not None:
            fig.suptitle(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"logitdiff_{token}.png"), bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0_token = list(tokens_of_interest.keys())[t0_idx]
            t1_token = list(tokens_of_interest.keys())[t1_idx]
            two_token_logitdiff_image = logitdiff_image[:, :, t0_idx] - logitdiff_image[:, :, t1_idx]
            abs_max = np.max(np.abs(two_token_logitdiff_image))

            # flatten ax
            fig, ax = plt.subplots(6, 6, figsize=(10, 12))
            ax = ax.flatten()

            for l in range(len(ax)):
                if l >= two_token_logitdiff_image.shape[0]:
                    ax[l].axis("off")
                else:
                    ax[l].imshow(np.asarray(image.convert("RGB").resize((image_size, image_size))))
                    logitdiff_map = two_token_logitdiff_image[l].reshape(image_size//patch_size, image_size//patch_size)
                    logitdiff_map_up = np.kron(logitdiff_map, np.ones((patch_size, patch_size)))
                    last_im = ax[l].imshow(logitdiff_map_up, cmap="RdBu", vmin=-abs_max, vmax=abs_max, alpha=0.8)
                    ax[l].set_xticks([])
                    ax[l].set_yticks([])
                    if cache_key == "resid_post":
                        if l == 0:
                            ax[l].set_title(f"L{l+1}: resid_pre")
                        else:
                            ax[l].set_title(f"L{l}")
                    else:
                        ax[l].set_title(f"L{l+1}")

            cbar = fig.colorbar(
                last_im, ax=ax, location="right",
                fraction=0.03,
                pad=0.02,
                aspect=30,
                shrink=0.9,
                label=f"logit difference ({t0_token} - {t1_token})"
            )
            if title is not None:
                fig.suptitle(title)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"logitdiff_{t0_token}_{t1_token}.png"), bbox_inches='tight', pad_inches=0.05)
            else:
                plt.show()
            plt.close()

def show_ablation_of_text_tokens(
    tokenizer,
    prompt: str,
    ablated_store: np.ndarray,
    original_store: np.ndarray,
    tokens_of_interest: dict[str, int],
    show_diff: list[int] = [[0, 1]],
    cache_key: str = "resid_post",
    save_dir: str = None,
    title: str = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    text_prompt_length: int = None,
    text_token_labels: list = None,
    ):
    # text_prompt_length/text_token_labels: pass the REAL post-image text-token
    # count and labels (from the processor-expanded input_ids); the fallback
    # below assumes 576 image tokens and img_token_id=32000, which is wrong for
    # anyres models (llava-v1.6/llava-next).
    
    assert cache_key in ["resid_post", "attn_out", "mlp_out", "resid_mid", "resid", "head_out"], f"Invalid cache key: {cache_key}"
    if cache_key == "head_out":
        raise NotImplementedError("head_out is not implemented yet")
    
    if text_prompt_length is not None:
        assert text_token_labels is not None and len(text_token_labels) == text_prompt_length
        text_prompt_token_list = list(text_token_labels)
    else:
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt)

        # Find the image token and replace it with image tokens
        img_token_count = (image_size // patch_size) ** 2  # 576 for 336x336 image with 14x14 patches

        token_labels = []
        for token_id in input_ids:
            if token_id == img_token_id:
                # One indexed because the HTML logic wants it that way
                token_labels.extend([f"<IMG{(i+1):03d}>" for i in range(img_token_count)])
            else:
                token_labels.append(tokenizer.decode([token_id]))

        text_prompt_length = len(token_labels)-(5+img_token_count) # len after <IMG>
        text_prompt_token_list = [token_labels[i] for i in range((len(token_labels)-text_prompt_length), len(token_labels))]
    

    logitdiff_text = ablated_store[:, -text_prompt_length:, :] - original_store[None, None, :] # (L, H, T)

    for i, token in enumerate(tokens_of_interest.keys()):
        plt.figure(figsize=(10, 10))
        abs_max = np.max(np.abs(logitdiff_text[:, :, i]))
        plt.imshow(logitdiff_text[:, :, i].T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
        plt.xlabel("layer")
        plt.ylabel("token")
        plt.yticks(range(len(text_prompt_token_list)), text_prompt_token_list, rotation=0)
        plt.colorbar(label=f"logit diff of {token}", shrink=0.5)
        if title is not None:
            plt.title(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"logitdiff_{token}.png"), bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0_token = list(tokens_of_interest.keys())[t0_idx]
            t1_token = list(tokens_of_interest.keys())[t1_idx]
            two_token_logitdiff_text = logitdiff_text[:, :, t0_idx] - logitdiff_text[:, :, t1_idx]
            abs_max = np.max(np.abs(two_token_logitdiff_text))

            plt.figure(figsize=(10, 10))
            plt.imshow(two_token_logitdiff_text.T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
            plt.xlabel("layer")
            plt.ylabel("token")
            plt.yticks(range(len(text_prompt_token_list)), text_prompt_token_list, rotation=0)
            plt.colorbar(label=f"logit diff ({t0_token} - {t1_token})", shrink=0.5)
            if title is not None:
                plt.title(title)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"logitdiff_{t0_token}_{t1_token}.png"), bbox_inches='tight', pad_inches=0.05)
            else:
                plt.show()
            plt.close()

def show_ablation_of_heads(
    tokenizer,
    prompt: str,
    ablated_store: np.ndarray,
    original_store: np.ndarray,
    tokens_of_interest: dict[str, int],
    show_diff: list[int] = [[0, 1]],
    # cache_key: str = "resid_post",
    save_dir: str = None,
    title: str = None,
    # image_size: int = 336,
    # patch_size: int = 14,
    # img_token_id: int = 32000,
    ):
    
    
    # # Tokenize the prompt

    # # Find the image token and replace it with image tokens

    #         # One indexed because the HTML logic wants it that way
    #         token_labels.extend([f"<IMG{(i+1):03d}>" for i in range(img_token_count)])
    #         token_labels.append(tokenizer.decode([token_id]))

    logitdiff_heads = ablated_store[:, :, :] - original_store[None, None, :] # (L, H, T)

    for i, token in enumerate(tokens_of_interest.keys()):
        plt.figure(figsize=(10, 10))
        abs_max = np.max(np.abs(logitdiff_heads[:, :, i]))
        plt.imshow(logitdiff_heads[:, :, i].T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
        plt.xlabel("layer")
        plt.ylabel("head")
        plt.colorbar(label=f"logit diff of {token}", shrink=0.5)
        if title is not None:
            plt.title(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"logitdiff_{token}.png"), bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0_token = list(tokens_of_interest.keys())[t0_idx]
            t1_token = list(tokens_of_interest.keys())[t1_idx]
            two_token_logitdiff_heads = logitdiff_heads[:, :, t0_idx] - logitdiff_heads[:, :, t1_idx]
            abs_max = np.max(np.abs(two_token_logitdiff_heads))

            plt.figure(figsize=(10, 10))
            plt.imshow(two_token_logitdiff_heads.T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
            plt.xlabel("layer")
            plt.ylabel("head")
            plt.colorbar(label=f"logit diff ({t0_token} - {t1_token})", shrink=0.5)
            if title is not None:
                plt.title(title)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"logitdiff_{t0_token}_{t1_token}.png"), bbox_inches='tight', pad_inches=0.05)
            else:
                plt.show()
            plt.close()
    
if __name__ == "__main__":
    from utils import *
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--image_name", type=str, default=None)
    parser.add_argument("--ablation_key", type=str, default=None)
    parser.add_argument("--images", type=str, default=None,
                        help="Override the hardcoded task list with cue-patching tasks (x-cue -> d-cue, "
                             "same pair as the va_s000 exemplar) for these stimuli: either a comma-separated "
                             "list of names or 'all-valid' for every manifest va_s* with a boundary for this model.")
    parser.add_argument("--chunk-idx", type=int, default=None,
                        help="Run tasks[chunk_idx::num_chunks] only. Each (image, ablation_key) pair writes its own "
                             "cache/plots, so chunks are independent; run the aggregate step after all chunks finish.")
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--prefix", type=str, default="I see a",
                        help="Assistant prefix for the --images cue-patching prompts. Cache and plot "
                             "directory slugs include it; the paper's figures use the default.")
    parser.add_argument("--text_only", action="store_true",
                        help="Patch only non-image token positions (~20x faster; the paper's "
                             "top_down_patching_every figures use text-token panels only).")

    args = parser.parse_args()
    model_id = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Expanded pipeline-generated set: stimuli live in data/duck_rabbit and
    # va_s* images are boundary-centered in memory (centered_va_image) instead
    # of reading the deprecated data/duck_rabbit_preprocessed files.
    bistable_image_dir = base_dir + '/data/duck_rabbit'
    with open(os.path.join(bistable_image_dir, "boundaries.json"), "r", encoding="utf-8") as f:
        boundaries = json.load(f)
    BOUNDARY_SLUG = slugify("List every animal in the image, each in one word." + "I see a")

    def load_stimulus(image_name, model_name):
        path = os.path.join(bistable_image_dir, f"{image_name}.png")
        if image_name.startswith("va_s"):
            img = centered_va_image(path, boundaries, model_name, image_name, BOUNDARY_SLUG)
            if img is None:
                raise ValueError(f"no boundary for {model_name}/{image_name}")
            return img
        return Image.open(path)

    if args.image_name is None:
        bistable_image_paths = [os.path.join(bistable_image_dir, f"{image_name}.png") for image_name in ["harper", "va_s000"]]
    else:
        bistable_image_paths = [os.path.join(bistable_image_dir, f"{args.image_name}.png")]
    results_dir = base_dir + '/outputs/resampling_ablation'
    os.makedirs(results_dir, exist_ok=True)
    
    if args.ablation_key is None:
        ablation_keys = ["resid_post", "attn_out", "mlp_out"]
    else:
        assert args.ablation_key in ["resid_post", "attn_out", "mlp_out"], f"Invalid ablation key: {args.ablation_key}"
        ablation_keys = [args.ablation_key]
    

    # Prompts
    source_prompts=[
        ["Name this animal in one word.", "It is a"], # prompt, prefix
        ["Name this animal in one word. It starts with 'd'.", "It is a"],
        ["Name this animal in one word. It starts with 'r'.", "It is a"],
        ["Is it a duck or a rabbit?", "It is a"],
        ["Is it a rabbit or a duck?", "It is a"],
        ["Name this hopping animal in one word.", "It is a"],
        ["Name this swimming animal in one word.", "It is a"],
        # ["Name this terrestrial animal in one word.", "It is a"],
        # ["Name this aquatic animal in one word.", "It is a"],
        # ["Is it a rabbit, a duck, both, or neither?"],
        # ["How many objects are in the image? Reply with a number only.", ""]
    ]
    target_prompts=[
        ["Name this animal in one word.", "It is a"], # prompt, prefix
        ["Name this animal in one word. It starts with 'd'.", "It is a"],
        ["Name this animal in one word. It starts with 'r'.", "It is a"],
        ["Is it a duck or a rabbit?", "It is a"],
        ["Is it a rabbit or a duck?", "It is a"],
        ["Name this hopping animal in one word.", "It is a"],
        ["Name this swimming animal in one word.", "It is a"],
        # ["Name this terrestrial animal in one word.", "It is a"],
        # ["Name this aquatic animal in one word.", "It is a"],
        # ["Is it a rabbit, a duck, both, or neither?"],
        # ["How many objects are in the image? Reply with a number only.", ""]
    ]


    # %%
    # Run only the model passed via --model. The previous hardcoded single-entry
    # dict silently ignored --model, so every SLURM array task ran llava-1.5-7b
    # (which is why only 7B plots ever existed for this experiment).
    vlm_dict = {args.model: VLM_DICT[args.model]}

    # %%
  # Skip (prompt, model) pairs whose results already exist.
  # Build the (prompt, model) task list.
    #     (image_path, prompt)
    #     # for vlm_name, model_id in vlm_dict.items()
    tasks = [ # (source_image_name, target_image_name, source_prompt, target_prompt)
        ("va_s000", "va_s000", ["List every animal in the image, each in one word. It starts with 'x'.", "It is a"], ["List every animal in the image, each in one word. It starts with 'd'.", "It is a"]),
        # ("va", "va", ["Name this animal in one word. It starts with 'd'.", "It is a"], ["Name this animal in one word. It starts with 'r'.", "It is a"]),
    ]

    if args.images is not None:
        # Same cue-patching pair as the exemplar, over an explicit stimulus set.
        if args.images == "all-valid":
            with open(os.path.join(bistable_image_dir, "va_set_manifest.json")) as f:
                names = [f"va_s{seed:03d}" for seed in json.load(f)["seeds"]]
        else:
            names = args.images.split(",")
        x_prompt = ["List every animal in the image, each in one word. It starts with 'x'.", args.prefix]
        d_prompt = ["List every animal in the image, each in one word. It starts with 'd'.", args.prefix]
        tasks = []
        for name in names:
            if name.startswith("va_s") and get_boundary(boundaries, model_id, name, BOUNDARY_SLUG) is None:
                print(f"[SKIP] no boundary for {model_id}/{name}")
                continue
            tasks.append((name, name, x_prompt, d_prompt))
        print(f"--images {args.images}: {len(tasks)} tasks for {model_id}")
        if args.chunk_idx is not None:
            assert args.num_chunks and 0 <= args.chunk_idx < args.num_chunks
            tasks = tasks[args.chunk_idx::args.num_chunks]
            print(f"[CHUNK {args.chunk_idx}/{args.num_chunks}] {len(tasks)} tasks", flush=True)

    if args.reverse:
        tasks = reversed(tasks)
    os.makedirs(os.path.join(results_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    for model_id in vlm_dict.keys():
        print(f"Running {model_id}...")
        processor, model, prompt_format = load_vlm(vlm_dict[model_id], return_prompt_format=True)
        tokenizer = processor.tokenizer
        # Per-model duck/rabbit continuation tokens (must stay entries [0,1]:
        # names must match the panel script's token names).
        duck_tok = {"llama3-llava-next-8b": "Ġduck"}.get(model_id, "▁du")
        rabbit_tok = {"llava-v1.6-mistral-7b": "▁rab",
                      "llama3-llava-next-8b": "Ġrabbit"}.get(model_id, "▁rabb")
        tokens_of_interest = {}
        for token in [duck_tok, rabbit_tok, "▁bird", "▁D", "▁Rab", "▁in", ".", "▁and"]:
            tid = tokenizer.convert_tokens_to_ids(token)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if tid is None or (unk_id is not None and tid == unk_id):
                alt = token.replace("▁", "Ġ") if "▁" in token else token.replace("Ġ", "▁")
                alt_id = tokenizer.convert_tokens_to_ids(alt)
                if alt_id is not None and alt_id != unk_id:
                    token, tid = alt, alt_id
                else:
                    continue  # token absent from this vocab; drop (mid-list only)
            tokens_of_interest[token] = tid
        for source_image_name, target_image_name, source_text_prompt, target_text_prompt in tqdm(tasks, desc='All tasks'):
            # Use each model's own chat template (mistral: [INST], llama3:
            # header tokens); the vicuna USER/ASSISTANT format is the fallback.
            if prompt_format is not None:
                source_prompt = prompt_format.format(prompt=source_text_prompt[0]) + source_text_prompt[1]
                target_prompt = prompt_format.format(prompt=target_text_prompt[0]) + target_text_prompt[1]
            else:
                source_prompt = f"USER: <image>\n{source_text_prompt[0]} ASSISTANT:{source_text_prompt[1]}"
                target_prompt = f"USER: <image>\n{target_text_prompt[0]} ASSISTANT:{target_text_prompt[1]}"
            source_prompt_slug = slugify(source_text_prompt[0]+source_text_prompt[1])
            target_prompt_slug = slugify(target_text_prompt[0]+target_text_prompt[1])
            source_image = load_stimulus(source_image_name, model_id)
            target_image = load_stimulus(target_image_name, model_id)
            cache_save_dir = os.path.join(results_dir, "cache", f"{source_image_name}_{target_image_name}_{source_prompt_slug}_{target_prompt_slug}_{model_id}")
            os.makedirs(cache_save_dir, exist_ok=True)

            ablated_store = model_resampling_ablation(model, processor, source_image, target_image, source_prompt, target_prompt, tokens_of_interest, device=device, save_dir=cache_save_dir, ablation_keys=ablation_keys, text_only=args.text_only)
            original_store = ablated_store["original"]
            # Real post-image text-token count/labels from the processor-expanded
            # input_ids (correct for anyres models too).
            _ids = processor(text=target_prompt, images=target_image, return_tensors="pt")["input_ids"][0].tolist()
            _img_tok_id = model.config.image_token_index
            _img_pos = [i for i, tid in enumerate(_ids) if tid == _img_tok_id]
            if _img_pos:
                n_post_image_text = len(_ids) - (_img_pos[-1] + 1)
                post_image_labels = [tokenizer.decode([tid]) for tid in _ids[-n_post_image_text:]]
            else:
                n_post_image_text, post_image_labels = None, None
            for component in ablation_keys:
                os.makedirs(os.path.join(results_dir, "plots", component, "image_tokens", model_id, f"{source_image_name}_{target_image_name}", f"{source_prompt_slug}_{target_prompt_slug}"), exist_ok=True)
                os.makedirs(os.path.join(results_dir, "plots", component, "text_tokens", model_id, f"{source_image_name}_{target_image_name}", f"{source_prompt_slug}_{target_prompt_slug}"), exist_ok=True)
                if not args.text_only:  # image-token rows are NaN in text_only stores
                    show_ablation_of_image_tokens(
                        tokenizer,
                        target_image,
                        ablated_store[component],
                        original_store,
                        tokens_of_interest,
                        show_diff=[[0, 1], [-3, -1]],
                        cache_key=component,
                        save_dir=os.path.join(results_dir, "plots", component, "image_tokens", model_id, f"{source_image_name}_{target_image_name}", f"{source_prompt_slug}_{target_prompt_slug}")
                    )
                show_ablation_of_text_tokens(
                    tokenizer,
                    target_prompt,
                    ablated_store[component],
                    original_store,
                    tokens_of_interest,
                    show_diff=[[0, 1], [-3, -1]],
                    cache_key=component,
                    save_dir=os.path.join(results_dir, "plots", component, "text_tokens", model_id, f"{source_image_name}_{target_image_name}", f"{source_prompt_slug}_{target_prompt_slug}"),
                    text_prompt_length=n_post_image_text,
                    text_token_labels=post_image_labels,
                )

        del model, processor
        torch.cuda.empty_cache()
