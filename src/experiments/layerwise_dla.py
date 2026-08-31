import glob
import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from utils import *
import math
import json
import torch
from PIL import Image
import torchvision.transforms.functional as TF
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


def _rmsnorm_apply_with_ref(norm_module, x, x_ref):
    eps = getattr(norm_module, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm_module, "eps", 1e-6)
    denom = torch.sqrt(x_ref.float().pow(2).mean(dim=-1, keepdim=True) + eps)  # [P,1]
    return (x / denom.to(dtype=x.dtype)) * norm_module.weight.to(dtype=x.dtype)

def _layernorm_apply_with_ref(norm_module, x, x_ref):
    eps = getattr(norm_module, "eps", 1e-5)
    mu = x_ref.mean(dim=-1, keepdim=True)
    var = (x_ref - mu).pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    y = (x - mu) * inv
    if hasattr(norm_module, "weight") and norm_module.weight is not None:
        y = y * norm_module.weight.to(dtype=x.dtype)
    if hasattr(norm_module, "bias") and norm_module.bias is not None:
        y = y + norm_module.bias.to(dtype=x.dtype)
    return y

def _apply_final_norm(norm_module, x, ln_mode: str, x_ref=None):
    if ln_mode == "none":
        return x
    if ln_mode == "independent":
        return norm_module(x)
    if ln_mode == "cached":
        if x_ref is None:
            raise ValueError("ln_mode='cached' requires ln_ref/x_ref.")
        is_rms = (
            hasattr(norm_module, "variance_epsilon")
            and hasattr(norm_module, "weight")
            and not hasattr(norm_module, "bias")
        )
        if is_rms or norm_module.__class__.__name__.lower().endswith("rmsnorm"):
            return _rmsnorm_apply_with_ref(norm_module, x, x_ref)
        else:
            return _layernorm_apply_with_ref(norm_module, x, x_ref)
    raise ValueError(f"Unknown ln_mode: {ln_mode}")

def logitlens(
    model,
    hidden_states,
    tokens_of_interest,
    pos,  # list[int]
    softmax=False,
    head_out=False,
    device="cpu",
    save_path=None,
    ln_mode: str = "independent",   # ["independent", "cached", "none"]
    ln_ref=None,               # required if ln_mode == "cached"
):
    """
    DLA-friendly logitlens by default (cached final norm stats).

    hidden_states:
      - not head_out: list/tuple length L, each [S,D] (or np arrays)
      - head_out: list/tuple length L, each [S,H,D] (or np arrays)
    ln_ref:
      - if ln_mode='cached': either [S,D] (single ref for all layers) OR [L,S,D] (per-layer ref)
        Typical: cache["resid_post"][-1]  (final full residual, pre-final-norm)
    """
    pos_t = torch.tensor(pos, device=device, dtype=torch.long)
    interesting_token_ids = torch.tensor(list(tokens_of_interest.values()), device=device, dtype=torch.long)

    norm = model.language_model.model.norm
    lm_head = model.language_model.lm_head

    # prepare ln_ref tensor if needed
    ln_ref_t = None
    if ln_mode == "cached":
        if ln_ref is None:
            raise ValueError("ln_mode='cached' is default, so please pass ln_ref (e.g., cache['resid_post'][-1]).")
        ln_ref_t = torch.from_numpy(ln_ref) if isinstance(ln_ref, np.ndarray) else ln_ref
        ln_ref_t = ln_ref_t.to(device)

    if not head_out:
        L = len(hidden_states)
        out = torch.zeros((L, len(pos_t), len(tokens_of_interest)), device=device, dtype=torch.float32)

        for l in range(L):
            x = hidden_states[l]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(device)                     # [S,D]
            x = x.index_select(0, pos_t)          # [P,D]

            if ln_mode == "cached":
                ref = ln_ref_t[l] if (ln_ref_t.ndim == 3 and ln_ref_t.shape[0] == L) else ln_ref_t
                if ref.ndim != 2:
                    raise ValueError(f"ln_ref must be [S,D] or [L,S,D], got {tuple(ref.shape)}")
                ref = ref.index_select(0, pos_t)  # [P,D]
                x_norm = _apply_final_norm(norm, x, "cached", x_ref=ref)
            else:
                x_norm = _apply_final_norm(norm, x, ln_mode)

            logits = lm_head(x_norm)  # [P,V]
            if softmax:
                logits = torch.softmax(logits, dim=-1)

            out[l] = logits.index_select(1, interesting_token_ids).float()

        out_np = out.detach().cpu().numpy()
        if save_path is not None:
            np.save(save_path, out_np)
        return out_np

    else:
        L = len(hidden_states)
        H = hidden_states[0].shape[1]
        out = torch.zeros((L, len(pos_t), H, len(tokens_of_interest)), device=device, dtype=torch.float32)

        for l in range(L):
            xh = hidden_states[l]
            if isinstance(xh, np.ndarray):
                xh = torch.from_numpy(xh)
            xh = xh.to(device)                  # [S,H,D]
            xh = xh.index_select(0, pos_t)       # [P,H,D]

            if ln_mode == "cached":
                ref = ln_ref_t[l] if (ln_ref_t.ndim == 3 and ln_ref_t.shape[0] == L) else ln_ref_t
                if ref.ndim != 2:
                    raise ValueError(f"ln_ref must be [S,D] or [L,S,D], got {tuple(ref.shape)}")
                ref = ref.index_select(0, pos_t)  # [P,D]
            else:
                ref = None

            for h in range(H):
                x = xh[:, h, :]  # [P,D]
                if ln_mode == "cached":
                    x_norm = _apply_final_norm(norm, x, "cached", x_ref=ref)
                else:
                    x_norm = _apply_final_norm(norm, x, ln_mode)

                logits = lm_head(x_norm)
                if softmax:
                    logits = torch.softmax(logits, dim=-1)
                out[l, :, h, :] = logits.index_select(1, interesting_token_ids).float()

        out_np = out.detach().cpu().numpy()
        if save_path is not None:
            np.save(save_path, out_np)
        return out_np

def dominant_patch_map_from_logitlens(
    logitlens_probs: np.ndarray,          # [L, P, T] from logitlens(..., softmax=True)
    tokens_of_interest: dict[str, int],   # keeps token ordering via keys()
    threshold: float = 0.1,
    img_pos: list[int] | np.ndarray | None = None,
    image_size: int = 336,
    patch_size: int = 14,
    pre_img_tokens: int = 5,              # your convention in show_logitlens_of_image_tokens
):
    """
    Returns:
      dominant_tokens: list[str|None] length P_img
      dominant_values: np.ndarray float (P_img,) with None-masked as np.nan (or you can return Python None list)
      per_token_max:   np.ndarray float (P_img, T)  (max over layers for each token)
      best_idx:        np.ndarray int   (P_img,)     (argmax over tokens)
    """
    assert logitlens_probs.ndim == 3, f"Expected [L,P,T], got {logitlens_probs.shape}"
    L, P, T = logitlens_probs.shape

    token_names = list(tokens_of_interest.keys())
    assert T == len(token_names), f"T mismatch: probs has T={T}, tokens_of_interest has {len(token_names)}"

    # Decide which positions are "image tokens"
    if img_pos is None:
        img_token_count = (image_size // patch_size) ** 2
        img_start = pre_img_tokens
        img_end = min(img_start + img_token_count, P)
        img_pos = np.arange(img_start, img_end)
    else:
        img_pos = np.asarray(img_pos)

    # [L, P_img, T]
    img_probs = logitlens_probs[:, img_pos, :].astype(np.float32)

    # For each (p_img, t): max over layers
    # [P_img, T]
    dominant_patch_map = img_probs.max(axis=0)
    below_threshold_map = dominant_patch_map < threshold
    dominant_patch_map[below_threshold_map] = 0

    return dominant_patch_map

def show_dominant_patch_map(
    dominant_patch_map: np.ndarray,
    image: Image.Image,
    tokens_of_interest: dict[str, int],
    save_dir: str = None,
    title: str = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    x_y_pairs: list[tuple[int, int]] = [[0, 1]],
):
    for idx, token in enumerate(list(tokens_of_interest.keys())):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(np.asarray(image.convert("L").resize((image_size, image_size))), cmap="gray")
        dominant_patch_map_up = np.kron(dominant_patch_map[..., idx].reshape(image_size//patch_size, image_size//patch_size), np.ones((patch_size, patch_size)))
        im = ax.imshow(dominant_patch_map_up, cmap="Blues", vmin=0, vmax=1, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(
            im, ax=ax, location="right",
            fraction=0.03,
            pad=0.02,
            aspect=30,
            shrink=0.9,
            label=f"max prob: {token}"
        )
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{token}.png"), bbox_inches='tight', pad_inches=0.05)

    for x_idx, y_idx in x_y_pairs:
        x_token = list(tokens_of_interest.keys())[x_idx]
        y_token = list(tokens_of_interest.keys())[y_idx]
        fig, ax = plt.subplots(figsize=(10, 10))
        dominant_patch_map_diff = dominant_patch_map[..., x_idx] - dominant_patch_map[..., y_idx]

        ax.imshow(np.asarray(image.convert("L").resize((image_size, image_size))), cmap="gray")
        dominant_patch_map_up = np.kron(dominant_patch_map_diff.reshape(image_size//patch_size, image_size//patch_size), np.ones((patch_size, patch_size)))
        im = ax.imshow(dominant_patch_map_up, cmap="RdBu", vmin=-1, vmax=1, alpha=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(
            im, ax=ax, location="right",
            fraction=0.03,
            pad=0.02,
            aspect=30,
            shrink=0.9,
            label=f"max prob diff: {x_token} - {y_token}"
        )
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{x_token}_{y_token}.png"), bbox_inches='tight', pad_inches=0.05)
        plt.close()

def plot_layer_trajectory(
    logits_of_interest: np.ndarray, # resid
    tokens_of_interest: dict[str, int],
    x_y_pairs: list[tuple[int, int]],
    cmap: str = "viridis",
    annotate_layers: bool = True,
    image: Image.Image = None,
    title: str = None,
    softmax: bool = False,
    save_dir: str = None,
    filename_prefix: str = "",
):
    """
    logits_of_interest: shape (num_layers, num_tokens)
    tokens: list of token strings in the same order as columns of logits_of_interest
    x_token, y_token: which two tokens to use for the x/y axes
    """
    for ix, iy in x_y_pairs:
        x_token = list(tokens_of_interest.keys())[ix]
        y_token = list(tokens_of_interest.keys())[iy]

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
                if layer_idx == 0:
                    annotation = "pre"
                elif layer_idx % 2 == 0:
                    annotation = f"{layer_idx//2 - 1}p"
                else:
                    annotation = f"{layer_idx//2}m"
                ax.annotate(
                    annotation,
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

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{filename_prefix}_{x_token}_{y_token}.png"), bbox_inches='tight', pad_inches=0.05)

        plt.close()

def show_logitlens_of_image_tokens(
    tokenizer,
    prompt: str,
    image: Image.Image,
    logitlens: np.ndarray,
    tokens_of_interest: dict[str, int],
    show_diff: list[int] = None,
    cache_key: str = "resid_post",
    save_dir: str = None,
    title: str = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    ):
    
    assert cache_key in ["resid_post", "attn_out", "mlp_out", "resid_mid", "resid", "head_out"], f"Invalid cache key: {cache_key}"
    assert show_diff is None or len(show_diff) == 2, f"show_diff must be a list of two integers: {show_diff}"
    if cache_key == "head_out":
        raise NotImplementedError("head_out is not implemented yet")
    
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

    for i, token in enumerate(tokens_of_interest.keys()):
        # logitlens for image tokens
        fig, ax = plt.subplots(6, 6, figsize=(10, 12))
        logitlens_image = logitlens[:, 5:-text_prompt_length, i]
        abs_max = np.max(np.abs(logitlens_image))

        # flatten ax
        ax = ax.flatten()

        for l in range(len(ax)):
            if l >= logitlens_image.shape[0]:
                ax[l].axis("off")
            else:
                ax[l].imshow(np.asarray(image.convert("L").resize((image_size, image_size))), cmap="gray")
                logitlens_map = logitlens_image[l].reshape(image_size//patch_size, image_size//patch_size)
                logitlens_map_up = np.kron(logitlens_map, np.ones((patch_size, patch_size)))
                last_im = ax[l].imshow(logitlens_map_up, cmap="RdBu", vmin=-abs_max, vmax=abs_max, alpha=0.8)
                ax[l].set_xticks([])
                ax[l].set_yticks([])
                if cache_key == "resid_post":
                    if l == 0:
                        ax[l].set_title("resid_pre")
                    else:
                        ax[l].set_title(f"L{l-1}")
                else:
                    ax[l].set_title(f"L{l}")

        cbar = fig.colorbar(
            last_im, ax=ax, location="right",
            fraction=0.03,
            pad=0.02,
            aspect=30,
            shrink=0.9,
            label=f"logit of {token}"
        )
        if title is not None:
            fig.suptitle(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"logitlens_{token}.png"), bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0_token = list(tokens_of_interest.keys())[t0_idx]
            t1_token = list(tokens_of_interest.keys())[t1_idx]
            logitdiff_image = logitlens[:, 5:-text_prompt_length, t0_idx] - logitlens[:, 5:-text_prompt_length, t1_idx]
            abs_max = np.max(np.abs(logitdiff_image))

            fig, ax = plt.subplots(6, 6, figsize=(10, 12))
            # flatten ax
            ax = ax.flatten()

            for l in range(len(ax)):
                if l >= logitdiff_image.shape[0]:
                    ax[l].axis("off")
                else:
                    ax[l].imshow(np.asarray(image.convert("L").resize((image_size, image_size))), cmap="gray")
                    logitlens_map = logitdiff_image[l].reshape(image_size//patch_size, image_size//patch_size)
                    logitlens_map_up = np.kron(logitlens_map, np.ones((patch_size, patch_size)))
                    last_im = ax[l].imshow(logitlens_map_up, cmap="RdBu", vmin=-abs_max, vmax=abs_max, alpha=0.8)
                    ax[l].set_xticks([])
                    ax[l].set_yticks([])
                    if cache_key == "resid_post":
                        if l == 0:
                            ax[l].set_title("resid_pre")
                        else:
                            ax[l].set_title(f"L{l-1}")
                    else:
                        ax[l].set_title(f"L{l}")

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

def show_logitlens_of_text_tokens(
    tokenizer,
    prompt: str,
    logitlens: np.ndarray,
    tokens_of_interest: dict[str, int],
    show_diff: list[int] = None,
    cache_key: str = "resid_post",
    save_dir: str = None,
    title: str = None,
    image_size: int = 336,
    patch_size: int = 14,
    img_token_id: int = 32000,
    ):
    
    assert cache_key in ["resid_post", "attn_out", "mlp_out", "resid_mid", "resid", "head_out"], f"Invalid cache key: {cache_key}"
    assert show_diff is None or len(show_diff) == 2, f"show_diff must be a list of two integers: {show_diff}"
    if cache_key == "head_out":
        raise NotImplementedError("head_out is not implemented yet")
    
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

    L = logitlens.shape[0]

    for i, token in enumerate(tokens_of_interest.keys()):
        plt.figure(figsize=(10, 10))
        logitlens_text = logitlens[:, -text_prompt_length:, i]
        abs_max = np.max(np.abs(logitlens_text))
        plt.imshow(logitlens_text.T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
        plt.xlabel("layer")
        plt.ylabel("token")
        if cache_key == "resid_post":
            ticks = [0] + (np.arange(0, L, 5) + 1).tolist()
            labels = ["pre"] + np.arange(0, L, 5).tolist()
            plt.xticks(ticks, labels, rotation=0)
        text_prompt_token_list = [token_labels[i] for i in range((len(token_labels)-text_prompt_length), len(token_labels))]
        plt.yticks(range(len(text_prompt_token_list)), text_prompt_token_list, rotation=0)
        plt.colorbar(label=f"logit of {token}", shrink=0.5)
        if title is not None:
            plt.title(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"logitlens_{token}.png"), bbox_inches='tight', pad_inches=0.05)
        else:
            plt.show()
        plt.close()

    if show_diff is not None:
        for t0_idx, t1_idx in show_diff:
            t0_token = list(tokens_of_interest.keys())[t0_idx]
            t1_token = list(tokens_of_interest.keys())[t1_idx]
            logitdiff_text = logitlens[:, -text_prompt_length:, t0_idx] - logitlens[:, -text_prompt_length:, t1_idx]
            abs_max = np.max(np.abs(logitdiff_text))

            plt.figure(figsize=(10, 10))
            plt.imshow(logitdiff_text.T, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
            plt.xlabel("layer")
            plt.ylabel("token")
            if cache_key == "resid_post":
                ticks = [0] + (np.arange(0, L, 5) + 1).tolist()
                labels = ["pre"] + np.arange(0, L, 5).tolist()
                plt.xticks(ticks, labels, rotation=0)
            text_prompt_token_list = [token_labels[i] for i in range((len(token_labels)-text_prompt_length), len(token_labels))]
            plt.yticks(range(len(text_prompt_token_list)), text_prompt_token_list, rotation=0)
            plt.colorbar(label=f"logit diff ({t0_token} - {t1_token})", shrink=0.5)
            if title is not None:
                plt.title(title)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"logitdiff_{t0_token}_{t1_token}.png"), bbox_inches='tight', pad_inches=0.05)
            else:
                plt.show()
            plt.close()
def show_dla_from_submodules(
    logitlens_cache_dirs: list[str],
    tokens_of_interest: dict[str, int],
    x_y_pairs: list[tuple[int, int]] = ((0, 1),),
    save_dir: str = None,
    title: str = None,
):
    """
    For each (x_idx, y_idx):
      - Final logit: resid_post_logitlens[-1, pos, token]
      - Attn out sum: sum_l attn_out_logitlens[l, pos, token]
      - MLP out sum:  sum_l mlp_out_logitlens[l, pos, token]

    Aggregates across cache dirs -> mean±std, then plots 3 groups × 2 bars.
    Uses pos=-1 (last position).
    """
    if len(logitlens_cache_dirs) == 0:
        raise ValueError("logitlens_cache_dirs is empty")

    token_names = list(tokens_of_interest.keys())
    T = len(token_names)
    x_y_pairs = [tuple(p) for p in x_y_pairs]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for pair_idx, (x_idx, y_idx) in enumerate(x_y_pairs):
        if not (0 <= x_idx < T and 0 <= y_idx < T):
            raise ValueError(f"x_y_pairs[{pair_idx}]={x_idx,y_idx} out of range for T={T}")

        x_tok = token_names[x_idx]
        y_tok = token_names[y_idx]

        # per-run scalars (we only store scalars; no big arrays)
        final_x, final_y = [], []
        attn_x, attn_y = [], []
        mlp_x, mlp_y = [], []

        for d in logitlens_cache_dirs:
            resid = np.load(os.path.join(d, "resid_post_logitlens.npy"))   # [L(+1), P, T]
            attn  = np.load(os.path.join(d, "attn_out_logitlens.npy"))     # [L, P, T]
            mlp   = np.load(os.path.join(d, "mlp_out_logitlens.npy"))      # [L, P, T]

            pos = -1  # last position

            # Final (use last resid entry)
            final_x.append(float(resid[-1, pos, x_idx]))
            final_y.append(float(resid[-1, pos, y_idx]))

            # Sum over layers
            attn_x.append(float(attn[:, pos, x_idx].sum()))
            attn_y.append(float(attn[:, pos, y_idx].sum()))

            mlp_x.append(float(mlp[:, pos, x_idx].sum()))
            mlp_y.append(float(mlp[:, pos, y_idx].sum()))

        def mean_std(xs: list[float]) -> tuple[float, float]:
            xs = np.asarray(xs, dtype=np.float32)
            m = float(xs.mean())
            s = float(xs.std(ddof=1)) if xs.size > 1 else 0.0
            return m, s

        mx_final, sx_final = mean_std(final_x)
        my_final, sy_final = mean_std(final_y)
        mx_attn,  sx_attn  = mean_std(attn_x)
        my_attn,  sy_attn  = mean_std(attn_y)
        mx_mlp,   sx_mlp   = mean_std(mlp_x)
        my_mlp,   sy_mlp   = mean_std(mlp_y)

        categories = ["Final logit", "Attn out", "MLP out"]
        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(
            x - width/2,
            [mx_final, mx_attn, mx_mlp],
            width,
            yerr=[sx_final, sx_attn, sx_mlp],
            capsize=4,
            label=x_tok,
        )
        ax.bar(
            x + width/2,
            [my_final, my_attn, my_mlp],
            width,
            yerr=[sy_final, sy_attn, sy_mlp],
            capsize=4,
            label=y_tok,
        )

        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Logit Value")
        if title is not None:
            ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"dla_submodules_{x_tok}_vs_{y_tok}.png")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            plt.close()
        else:
            plt.show()
            plt.close()

def _infer_img_token_id(processor, model, img_token_id: Optional[int]) -> int:
    if img_token_id is not None:
        return img_token_id
    # try config first (common for LLaVA)
    v = getattr(model.config, "image_token_index", None)
    if v is None and hasattr(model, "language_model"):
        v = getattr(model.language_model.config, "image_token_index", None)
    if v is not None:
        return int(v)
    # fallback: tokenizer
    tok = processor.tokenizer
    v = tok.convert_tokens_to_ids("<image>")
    if v == tok.unk_token_id:
        raise ValueError("Could not infer img_token_id: '<image>' is not a known token.")
    return int(v)

def zero_ablate_image_embed_with_dominant_patch_map(
    model,
    processor,
    image,
    prompt,
    tokens_of_interest,
    dominant_patch_map,                  # (P_img, T), non-zero = selected mask
    device="cpu",
    save_dir=None,
    img_token_id: Optional[int] = 32000, # pass None to infer via _infer_img_token_id
    random_n: int = 10,
    seed: int = 0,
):
    """
    Saves (if save_dir):
      - original.npy              (T,)
      - zero_ablate_{t}.npy       (T,)
      - random_ablate_{t}.npy     (random_n, T)
    """
    base_dtype = next(model.parameters()).dtype
    llm = model.language_model.model

    token_names = list(tokens_of_interest.keys())
    interesting_ids = torch.tensor(list(tokens_of_interest.values()), device=device, dtype=torch.long)
    T = len(token_names)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # --- preprocess once ---
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_ids_1d = inputs["input_ids"][0]  # [S]
    S = int(input_ids_1d.numel())

    # --- infer image token id ---
    img_token_id_eff = _infer_img_token_id(processor, model, img_token_id)

    # --- positions of image tokens in the sequence ---
    img_pos = (input_ids_1d == img_token_id_eff).nonzero(as_tuple=False).flatten()  # [P_img_seq]
    if img_pos.numel() == 0:
        raise RuntimeError(
            f"No image tokens found in input_ids with img_token_id={img_token_id_eff}. "
            "If your model uses a single <image> token with internal expansion, this hook point won't work."
        )

    # --- dominant map: (P_img, T) ---
    dom = np.asarray(dominant_patch_map)
    if dom.ndim != 2:
        raise ValueError(f"dominant_patch_map must be 2D (P_img,T), got {dom.shape}")
    if dom.shape[1] != T:
        raise ValueError(f"dominant_patch_map second dim must be T={T}, got {dom.shape[1]}")

    # align dom rows with img_pos length (common off-by-one)
    if dom.shape[0] == img_pos.numel():
        pass
    elif dom.shape[0] + 1 == img_pos.numel():
        img_pos = img_pos[1:]
    elif dom.shape[0] == img_pos.numel() + 1:
        dom = dom[1:, :]
    else:
        raise ValueError(
            f"Length mismatch: dominant_patch_map has {dom.shape[0]} rows but found {img_pos.numel()} image tokens."
        )

    P_img = dom.shape[0]

    # --- forward helper: probs over vocab -> select tokens_of_interest ---
    def run_forward_probs_of_interest():
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]                 # [B,S,D]
            logits = model.language_model.lm_head(last_hidden)  # [B,S,V]
            probs = torch.softmax(logits[0, -1, :], dim=-1)     # [V]
            return probs.index_select(0, interesting_ids).detach().cpu().numpy().astype(np.float32)

    # --- single hook: multiply by binary keep-mask over sequence positions ---
    state = {"keep_mask_seq": None}  # Tensor [S] on device, dtype matches hidden_states

    def _pre_hook(module, inp):
        keep = state["keep_mask_seq"]
        if keep is None:
            return inp
        x = inp[0]  # [B,S,D]
        # broadcast [S] -> [B,S,1]
        x_new = x * keep.view(1, -1, 1)
        return (x_new,) + inp[1:]

    handle = llm.layers[0].register_forward_pre_hook(_pre_hook)

    try:
        results = {}

        # original
        state["keep_mask_seq"] = None
        original = run_forward_probs_of_interest()
        results["original"] = original
        if save_dir is not None:
            np.save(os.path.join(save_dir, "original.npy"), original)

        rng = np.random.default_rng(seed)
        img_pos_cpu = img_pos.detach().cpu().numpy()

        # per token: targeted + random baselines
        for t in range(T):
            # --- targeted mask from dominant map (binary) ---
            # ablate image token i iff dom[i, t] != 0
            idx_img = np.where(dom[:, t] != 0)[0]   # indices in [0..P_img-1]
            K = int(idx_img.size)

            # build keep mask over *sequence* positions
            if K == 0:
                ablated = original.copy()
            else:
                keep = torch.ones(S, device=device, dtype=base_dtype)
                seq_positions = img_pos[torch.tensor(idx_img, device=device, dtype=torch.long)]
                keep[seq_positions] = 0.0
                state["keep_mask_seq"] = keep
                ablated = run_forward_probs_of_interest()

            results[f"zero_ablate_{t}"] = ablated
            if save_dir is not None:
                np.save(os.path.join(save_dir, f"zero_ablate_{t}.npy"), ablated)

            # --- random baseline: ablate same K image tokens, repeated random_n times ---
            rand_store = np.empty((random_n, T), dtype=np.float32)
            if K == 0:
                rand_store[:] = original[None, :]
            else:
                K_eff = min(K, P_img)
                for r in range(random_n):
                    chosen_img = rng.choice(P_img, size=K_eff, replace=False)  # indices in image-token space
                    keep = torch.ones(S, device=device, dtype=base_dtype)
                    seq_positions = torch.tensor(img_pos_cpu[chosen_img], device=device, dtype=torch.long)
                    keep[seq_positions] = 0.0
                    state["keep_mask_seq"] = keep
                    rand_store[r] = run_forward_probs_of_interest()

            results[f"random_ablate_{t}"] = rand_store
            if save_dir is not None:
                np.save(os.path.join(save_dir, f"random_ablate_{t}.npy"), rand_store)

        # reset
        state["keep_mask_seq"] = None
        return results

    finally:
        handle.remove()

def plot_zero_ablation_effect_with_dominant_patch_map(
    zero_ablation_cache_dirs: list[str],
    tokens_of_interest: dict[str, int],
    x_y_pairs: list[tuple[int, int]] = ((0, 1),),
    save_dir: str = None,
    title: str = None,
):
    """
    Each cache dir should contain:
      - original.npy              (T,)
      - zero_ablate_{i}.npy       (T,)
      - random_ablate_{i}.npy     (R,T)   (optional)

    For each (x_idx,y_idx), makes ONE plot showing BOTH P(x) (blue) and P(y) (orange)
    over [x ablated, original, y ablated], with per-image faint lines + mean±std overlay,
    and dotted random baseline overlays if available.
    """
    if len(zero_ablation_cache_dirs) == 0:
        raise ValueError("zero_ablation_cache_dirs is empty")

    token_names = list(tokens_of_interest.keys())
    T = len(token_names)
    x_y_pairs = [tuple(p) for p in x_y_pairs]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def mean_std_over_images(curves: np.ndarray):
        # curves: (N,3)
        m = curves.mean(axis=0)
        s = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(m)
        return m, s

    def pooled_mean_std(chunks_1d_list):
        # chunks list of arrays (R,) possibly from multiple images -> pool them
        if len(chunks_1d_list) == 0:
            return None
        v = np.concatenate(chunks_1d_list, axis=0)
        m = float(v.mean())
        s = float(v.std(ddof=1)) if v.size > 1 else 0.0
        return m, s

    for pair_i, (x_idx, y_idx) in enumerate(x_y_pairs):
        if not (0 <= x_idx < T and 0 <= y_idx < T):
            raise ValueError(f"x_y_pairs[{pair_i}]={x_idx,y_idx} out of range for T={T}")

        x_tok = token_names[x_idx]
        y_tok = token_names[y_idx]

        xs = np.arange(3)
        xticklabels = [
            f"ablated with {x_tok} logitlens",
            "original",
            f"ablated with {y_tok} logitlens",
        ]

        # per-image curves (store only 3 numbers per image per token)
        px_curves = []  # P(x): (N,3)
        py_curves = []  # P(y): (N,3)

        # pooled random baselines (each gives a distribution at an endpoint)
        # - random_ablate_{x_idx} corresponds to endpoint 0
        # - random_ablate_{y_idx} corresponds to endpoint 2
        rand_x_for_px, rand_x_for_py = [], []
        rand_y_for_px, rand_y_for_py = [], []

        for d in zero_ablation_cache_dirs:
            orig = np.load(os.path.join(d, "original.npy")).astype(np.float32)               # (T,)
            xabl = np.load(os.path.join(d, f"zero_ablate_{x_idx}.npy")).astype(np.float32)  # (T,)
            yabl = np.load(os.path.join(d, f"zero_ablate_{y_idx}.npy")).astype(np.float32)  # (T,)

            px = np.array([xabl[x_idx], orig[x_idx], yabl[x_idx]], dtype=np.float32)
            py = np.array([xabl[y_idx], orig[y_idx], yabl[y_idx]], dtype=np.float32)

            px_curves.append(px)
            py_curves.append(py)

            # random baselines
            rx_path = os.path.join(d, f"random_ablate_{x_idx}.npy")
            ry_path = os.path.join(d, f"random_ablate_{y_idx}.npy")
            if os.path.exists(rx_path):
                rx = np.load(rx_path).astype(np.float32)  # (R,T)
                rand_x_for_px.append(rx[:, x_idx])
                rand_x_for_py.append(rx[:, y_idx])
            if os.path.exists(ry_path):
                ry = np.load(ry_path).astype(np.float32)  # (R,T)
                rand_y_for_px.append(ry[:, x_idx])
                rand_y_for_py.append(ry[:, y_idx])

        px_curves = np.stack(px_curves, axis=0)  # (N,3)
        py_curves = np.stack(py_curves, axis=0)  # (N,3)

        px_m, px_s = mean_std_over_images(px_curves)
        py_m, py_s = mean_std_over_images(py_curves)

        fig, ax = plt.subplots(figsize=(10, 4.8))

        # --- per-image faint lines (same style for all samples) ---
        for i in range(px_curves.shape[0]):
            ax.plot(xs, px_curves[i], marker="o", linewidth=1.0, alpha=0.25, color="0.5")
            ax.plot(xs, py_curves[i], marker="o", linewidth=1.0, alpha=0.25, color="0.5")

        # --- mean ± std (colored) ---
        ax.errorbar(xs, px_m, yerr=px_s, marker="o", linewidth=2.5, capsize=4, label=f"P({x_tok})", color="tab:blue")
        ax.errorbar(xs, py_m, yerr=py_s, marker="o", linewidth=2.5, capsize=4, label=f"P({y_tok})", color="tab:orange")

        # --- random baseline dotted overlays (colored, with endpoint errorbars) ---
        # endpoint 0 baseline from random_ablate_{x_idx}
        rx_px = pooled_mean_std(rand_x_for_px)
        rx_py = pooled_mean_std(rand_x_for_py)
        if rx_px is not None and rx_py is not None:
            # draw dotted lines using baseline endpoints + original mean as middle anchor
            ax.plot([0, 1], [rx_px[0], px_m[1]], linestyle=":", linewidth=2.0, color="tab:blue", label=f"P({x_tok}) random baseline")
            ax.plot([0, 1], [rx_py[0], py_m[1]], linestyle=":", linewidth=2.0, color="tab:orange", label=f"P({y_tok}) random baseline")
            ax.errorbar([0], [rx_px[0]], yerr=[rx_px[1]], marker="s", capsize=4, linewidth=0, color="tab:blue")
            ax.errorbar([0], [rx_py[0]], yerr=[rx_py[1]], marker="s", capsize=4, linewidth=0, color="tab:orange")

        # endpoint 2 baseline from random_ablate_{y_idx}
        ry_px = pooled_mean_std(rand_y_for_px)
        ry_py = pooled_mean_std(rand_y_for_py)
        if ry_px is not None and ry_py is not None:
            ax.plot([1, 2], [px_m[1], ry_px[0]], linestyle=":", linewidth=2.0, color="tab:blue")
            ax.plot([1, 2], [py_m[1], ry_py[0]], linestyle=":", linewidth=2.0, color="tab:orange")
            ax.errorbar([2], [ry_px[0]], yerr=[ry_px[1]], marker="s", capsize=4, linewidth=0, color="tab:blue")
            ax.errorbar([2], [ry_py[0]], yerr=[ry_py[1]], marker="s", capsize=4, linewidth=0, color="tab:orange")

        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(xticklabels, rotation=0)
        ax.set_ylabel("Probability")
        ax.set_title(title if title is not None else f"Zero-ablation effect (dominant map): {x_tok} vs {y_tok}")

        # Keep legend compact
        ax.legend(loc="best", fontsize=9)

        plt.tight_layout()

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"zero_ablation_effect_{x_tok}_vs_{y_tok}.png")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            plt.close()
        else:
            plt.show()
            plt.close()

def _dataset_item_to_pil_image(sample):
    """
    Robustly extract a PIL image from a RotatedRGBAImageDataset item.
    """
    x = sample

    if isinstance(sample, dict):
        if "image" in sample:
            x = sample["image"]
        else:
            # fallback: first value
            x = next(iter(sample.values()))
    elif isinstance(sample, (tuple, list)):
        x = sample[0]

    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x):
        return TF.to_pil_image(x.cpu())
    if isinstance(x, np.ndarray):
        return Image.fromarray(x)

    raise TypeError(f"Unsupported dataset sample type for image extraction: {type(x)}")


def make_centered_analysis_image(image_path, center_angle: float):
    """
    Build a single centered image using RotatedRGBAImageDataset, matching
    the center-angle logic used in top_down_cues.
    """
    dataset = RotatedRGBAImageDataset(str(image_path), [center_angle])
    if len(dataset) == 0:
        raise RuntimeError(f"RotatedRGBAImageDataset returned empty dataset for {image_path} @ {center_angle}")
    sample = dataset[0]
    return _dataset_item_to_pil_image(sample)


def collect_layerwise_dla_curves_from_cache(
    model,
    processor,
    image,
    text_input: str,
    cache: dict,
    tokens_of_interest: dict[str, int],   # {"▁bird":..., "▁rabb":...}
    cache_save_dir: str | None = None,
    device="cpu",
    img_token_id: int = 32000,
    use_saved_files: bool = False,
):
    """
    Uses an already-loaded cache and returns dict of arrays, each [L, T] at the LAST position only:
      - resid_post
      - attn_out
      - mlp_out
      - attn_out_image_only

    resid_post excludes resid_pre, so all returned arrays have the same first dimension L.
    """
    curves_npz_path = None
    if cache_save_dir is not None:
        os.makedirs(cache_save_dir, exist_ok=True)
        curves_npz_path = os.path.join(cache_save_dir, "layerwise_dla_last_bird_rabb.npz")

    if use_saved_files and curves_npz_path is not None and os.path.exists(curves_npz_path):
        data = np.load(curves_npz_path)
        return {
            "resid_post": data["resid_post"],
            "attn_out": data["attn_out"],
            "mlp_out": data["mlp_out"],
            "attn_out_image_only": data["attn_out_image_only"],
        }

    last_pos = int(cache["resid_post"].shape[1] - 1)

    # resid_post: drop resid_pre so length matches L
    resid_post_logitlens = logitlens(
        model=model,
        hidden_states=cache["resid_post"][1:],   # [L,S,D]
        tokens_of_interest=tokens_of_interest,
        pos=[last_pos],
        device=device,
        ln_mode="cached",
        ln_ref=cache["resid_post"][-1],
        save_path=(os.path.join(cache_save_dir, "resid_post_no_pre_logitlens_last.npy") if (use_saved_files and cache_save_dir is not None) else None),
    )[:, 0, :]  # [L,T]

    attn_out_logitlens = logitlens(
        model=model,
        hidden_states=cache["attn_out"],         # [L,S,D]
        tokens_of_interest=tokens_of_interest,
        pos=[last_pos],
        device=device,
        ln_mode="cached",
        ln_ref=cache["resid_post"][-1],
        save_path=(os.path.join(cache_save_dir, "attn_out_logitlens_last.npy") if (use_saved_files and cache_save_dir is not None) else None),
    )[:, 0, :]  # [L,T]

    mlp_out_logitlens = logitlens(
        model=model,
        hidden_states=cache["mlp_out"],          # [L,S,D]
        tokens_of_interest=tokens_of_interest,
        pos=[last_pos],
        device=device,
        ln_mode="cached",
        ln_ref=cache["resid_post"][-1],
        save_path=(os.path.join(cache_save_dir, "mlp_out_logitlens_last.npy") if (use_saved_files and cache_save_dir is not None) else None),
    )[:, 0, :]  # [L,T]

    # image-only attention output DLA
    image_only_head_out_path = (
        os.path.join(cache_save_dir, "image_only_head_out_dla_last.npy")
        if (use_saved_files and cache_save_dir is not None)
        else None
    )

    head_out_image_only = collect_knockout_head_out_dla_by_layer(
        model=model,
        processor=processor,
        image=image,
        prompt=text_input,
        device=device,
        attention_knockout_key="image",
        attention_knockout_query="last",
        attention_knockout_mode="keep_only",
        attention_knockout_stage="post_softmax",
        img_token_id=img_token_id,
        tokens_of_interest=tokens_of_interest,
        pos=[last_pos],
        save_path=image_only_head_out_path,
    )  # [L,H,1,T]

    attn_out_image_only = head_out_image_only[:, :, 0, :].sum(axis=1)  # [L,T]

    if curves_npz_path is not None and use_saved_files:
        np.savez(
            curves_npz_path,
            resid_post=resid_post_logitlens,
            attn_out=attn_out_logitlens,
            mlp_out=mlp_out_logitlens,
            attn_out_image_only=attn_out_image_only,
        )

    return {
        "resid_post": resid_post_logitlens,
        "attn_out": attn_out_logitlens,
        "mlp_out": mlp_out_logitlens,
        "attn_out_image_only": attn_out_image_only,
    }

def plot_layerwise_dla_mean_std_one_panel(
    curves_by_image: list[dict[str, np.ndarray]],
    tokens_of_interest: dict[str, int],
    save_path: str | None = None,
    title: str | None = None,
):
    """
    curves_by_image: list of dicts returned by collect_layerwise_dla_curves_for_input
    Each entry has arrays of shape [L,T].

    Produces one panel with:
      colors = component
      linestyle = token
      mean ± std across images
    """
    if len(curves_by_image) == 0:
        raise ValueError("curves_by_image is empty")

    token_names = list(tokens_of_interest.keys())
    if len(token_names) != 2:
        raise ValueError(f"Expected exactly 2 tokens, got {token_names}")

    #     ("resid_post", "resid_post", "black"),
    #     ("attn_out", "attn_out", "tab:blue"),
    #     ("attn_out_image_only", "attn_out (image only)", "tab:orange"),
    #     ("mlp_out", "mlp_out", "tab:green"),
    #     token_names[0]: "-",
    #     token_names[1]: "--",

    component_order = [
        ("resid_post", "resid_post"),
        ("attn_out", "attn_out"),
        ("attn_out_image_only", "attn_out (image only)"),
        ("mlp_out", "mlp_out"),
    ]

    component_styles = {
        "resid_post": {"linestyle": "-",  "linewidth": 2.6},
        "attn_out": {"linestyle": ":",  "linewidth": 2.6},
        "attn_out_image_only": {"linestyle": ":",  "linewidth": 2.2},
        "mlp_out": {"linestyle": "--", "linewidth": 2.4},
    }

    component_token_colors = {
        "resid_post": {
            "▁bird": "tab:blue",
            "▁rabb": "tab:orange",
        },
        "attn_out": {
            "▁bird": "tab:blue",
            "▁rabb": "tab:orange",
        },
        "attn_out_image_only": {
            "▁bird": "deepskyblue",
            "▁rabb": "red",
        },
        "mlp_out": {
            "▁bird": "royalblue",
            "▁rabb": "darkorange",
        },
    }

    # stack -> [N,L,T]
    stacked = {}
    for key, _ in component_order:
        stacked[key] = np.stack([d[key] for d in curves_by_image], axis=0)

    L = stacked["resid_post"].shape[1]
    layers = np.arange(L)

    fig, ax = plt.subplots(figsize=(8, 8))

    for key, label in component_order:
        arr = stacked[key]  # [N,L,T]
        mean = arr.mean(axis=0)  # [L,T]
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

        style = component_styles[key]

        for ti, tok in enumerate(token_names):
            # tokens differ across tokenizers (e.g. Ġbird); fall back by position
            color = component_token_colors[key].get(
                tok, ["tab:blue", "tab:orange", "tab:green"][ti % 3])
            y = mean[:, ti]
            s = std[:, ti]

            ax.plot(
                layers,
                y,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                color=color,
                label=f"{label} | {tok}",
            )
            ax.fill_between(
                layers,
                y - s,
                y + s,
                color=color,
                alpha=0.08,
                linewidth=0,
            )

    ax.axhline(0.0, linewidth=1, color="gray")
    ax.set_xlabel("layer")
    ax.set_ylabel("direct logit attribution")
    ax.set_xlim(0, L - 1)
    ax.set_xticks(np.arange(0, L, 5))
    if title is not None:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--image_name", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading/inference; rebuild the layerwise-DLA mean/std panels from cached per-image curves (CPU-friendly).")
    args = parser.parse_args()
    model_id = args.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bistable_image_dir = os.path.join(DATA_DIR, 'duck_rabbit')
    if args.image_name is None:
        # The Visual Anagram set (see data/duck_rabbit/va_set_manifest.json).
        bistable_image_paths = [os.path.join(bistable_image_dir, f"{image_name}.png") for image_name in ["harper"]]
        bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "va_s*.png")))
    else:
        bistable_image_paths = [os.path.join(bistable_image_dir, f"{args.image_name}.png")]
    results_dir = os.path.join(OUTPUTS_DIR, 'layerwise_dla')
    os.makedirs(results_dir, exist_ok=True)
    

    prompts = [
        ["List every animal in the image, each in one word.", "I see a"],
        # ["List every animal in the image, each in one word.", "There's a"],
        # ["List every animal in the image, each in one word.", "There's a duck"],
        # ["List every animal in the image, each in one word.", "There's a rabbit"],
        # # ["List every animal in the image, each in one word.", "There's a bird"],
        # ["List every animal in the image, each in one word. It starts with 'd'.", "There's a"],
        # ["List every animal in the image, each in one word. It starts with 'r'.", "There's a"],
    ]

    # %%
    # Run only the model passed via --model (SLURM array passes each LLaVA model).
    vlm_dict = {args.model: VLM_DICT[args.model]}

    # %%
  # Skip (prompt, model) pairs whose results already exist.
  # Build the (prompt, model) task list.
    tasks = [
        (image_path, prompt)
        for image_path in bistable_image_paths
        for prompt in prompts
        # for vlm_name, model_id in vlm_dict.items()
    ]
    os.makedirs(os.path.join(results_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    boundaries_path = bistable_image_dir + "/boundaries.json"
    if os.path.exists(boundaries_path):
        with open(boundaries_path, "r", encoding="utf-8") as f:
            boundaries = json.load(f)
    else:
        boundaries = {}
        print(f"[WARN] boundaries file not found: {boundaries_path}")

    from collections import defaultdict

    # aggregate over valid visual anagrams only
    curves_by_prompt_slug = defaultdict(list)
    included_images_by_prompt_slug = defaultdict(list)
    prompt_meta_by_slug = {}

    for model_id in vlm_dict.keys():
        print(f"Running {model_id}...")
        if args.plot_only:
            processor = model = prompt_format = tokenizer = None
            print("[PLOT-ONLY] skipping model load; rebuilding DLA panels from cached curves")
        else:
            processor, model, prompt_format = load_vlm(vlm_dict[model_id], return_prompt_format=True)
            model.to(device)
            model.eval()
            tokenizer = processor.tokenizer

        def adapt_token(token_str):
            """Swap SentencePiece '▁' vs BPE 'Ġ' word-boundary prefix to match
            this model's vocab (llama3-llava-next-8b uses GPT-2-style BPE)."""
            tid = tokenizer.convert_tokens_to_ids(token_str)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if tid is None or (unk_id is not None and tid == unk_id):
                alt = token_str.replace("▁", "Ġ") if "▁" in token_str else token_str.replace("Ġ", "▁")
                alt_tid = tokenizer.convert_tokens_to_ids(alt)
                if alt_tid is not None and alt_tid != unk_id:
                    return alt
            return token_str

        # Adapt per-model and drop tokens absent from this model's vocab.
        tokens_of_interest = {}
        for token in ([] if args.plot_only else ["▁bird", "▁rabb", "▁du", "▁b", "▁D", "▁Rab", "▁in", ".", "▁and"]):
            tok = adapt_token(token)
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                tokens_of_interest[tok] = tid
        # Defined up front (plot-only runs may skip every image and must still
        # reach the aggregation below without a NameError).
        _bird_tok0, _rabb_tok0 = get_duck_rabbit_tokens(model_id)
        tokens_of_interest_bird_rabb = {
            token: (None if args.plot_only else tokenizer.convert_tokens_to_ids(token))
            for token in [_bird_tok0, _rabb_tok0]
        }
        for image_path, text_prompt in tqdm(tasks, desc='All tasks'):
            image_name = os.path.basename(image_path).split('.')[0]
            prompt, prompt_prefix = text_prompt[0], text_prompt[1]
            if prompt_format is not None:
                text_input = prompt_format.format(prompt=prompt) + prompt_prefix
            else:
                text_input = f"USER: <image>\n{prompt} ASSISTANT:{prompt_prefix}"
            prompt_slug = slugify(prompt+prompt_prefix)
            prompt_slug_for_boundary = prompt_slug

            if image_name.startswith("va_"):
                center_angle = get_boundary(
                    boundaries=boundaries,
                    model_name=model_id,
                    image_name=image_name,
                    prompt_slug=prompt_slug_for_boundary,
                )
                if center_angle is None:
                    print(f"[SKIP] No boundary for {model_id} | {image_name} | {prompt_slug_for_boundary}")
                    continue
            else:
                center_angle = 0

            image = make_centered_analysis_image(image_path, center_angle)

            cache_save_dir = os.path.join(results_dir, "cache", model_id, image_name, prompt_slug)
            os.makedirs(cache_save_dir, exist_ok=True)

            cache = None if args.plot_only else get_cache(model, processor, image, text_input, device=device, save_dir=cache_save_dir, cache_keys=["resid_post", "attn_out", "mlp_out"])
            if args.plot_only and not os.path.exists(os.path.join(cache_save_dir, "layerwise_dla_last_bird_rabb.npz")):
                print(f"[PLOT-ONLY SKIP] missing cached curves: {cache_save_dir}")
                continue


            # Per-model duck/rabbit token pair (mistral: ▁rab, llama3: Ġbird/Ġrabbit).
            _bird_tok, _rabb_tok = get_duck_rabbit_tokens(model_id)
            tokens_of_interest_bird_rabb = {
                token: (None if args.plot_only else tokenizer.convert_tokens_to_ids(token))
                for token in [_bird_tok, _rabb_tok]
            }
            curves = collect_layerwise_dla_curves_from_cache(
                model=model,
                processor=processor,
                image=image,
                text_input=text_input,
                cache=cache,
                tokens_of_interest=tokens_of_interest_bird_rabb,
                cache_save_dir=cache_save_dir,
                device=device,
                # llama3-llava-next uses a different <image> token id than the
                # 32000 default (vicuna/mistral tokenizers).
                img_token_id=(32000 if args.plot_only else getattr(model.config, "image_token_index", 32000)),
                use_saved_files=True,
            )

            # aggregate only over valid visual anagrams
            if image_name.startswith("va_"):
                curves_by_prompt_slug[prompt_slug].append(curves)
                included_images_by_prompt_slug[prompt_slug].append(image_name)
                prompt_meta_by_slug[prompt_slug] = {
                    "prompt": prompt,
                    "prompt_prefix": prompt_prefix,
                }
        

            for component in ([] if args.plot_only else ["resid_post", "mlp_out", "attn_out"]):
                logits_of_interest = logitlens(
                    model, cache[component], tokens_of_interest, 
                    pos=[t for t in range(cache[component].shape[-2])], 
                    device=device, 
                    save_path=os.path.join(cache_save_dir, f"{component}_logitlens.npy"),
                    ln_mode="cached",
                    ln_ref=cache["resid_post"][-1],
                )

                img_results_dir = os.path.join(results_dir, "plots", "image_tokens", component, model_id, image_name) # independent from prompt
                txt_results_dir = os.path.join(results_dir, "plots", "text_tokens", component, model_id, image_name, prompt_slug)
                dominant_patch_map_dir = os.path.join(results_dir, "plots", "dominant_patch_map", model_id, image_name)
                os.makedirs(img_results_dir, exist_ok=True)
                os.makedirs(txt_results_dir, exist_ok=True)
                os.makedirs(dominant_patch_map_dir, exist_ok=True)

                if component == "resid":
                    final_token_trajectory_dir = os.path.join(results_dir, "plots", "final_token_trajectory", model_id, image_name)
                    os.makedirs(final_token_trajectory_dir, exist_ok=True)
                    plot_layer_trajectory(
                        logits_of_interest[..., -1, :],
                        tokens_of_interest=tokens_of_interest,
                        x_y_pairs=[
                            [0, 1], # bird vs rabbit
                            [-3, -1], # in vs and
                            ],
                        image=image,
                        title=f"({model_id})\n{prompt} prefix: {prompt_prefix}",
                        save_dir=final_token_trajectory_dir,
                        filename_prefix=f"{prompt_slug}",
                    )
                elif component == "resid_post":
                    # Per-model duck/rabbit token pair (mistral: ▁rab, llama3: Ġbird/Ġrabbit).
                    _bird_tok, _rabb_tok = get_duck_rabbit_tokens(model_id)
                    tokens_of_interest_bird_rabb = {token: tokenizer.convert_tokens_to_ids(token) for token in [_bird_tok, _rabb_tok]}
                    logits_of_interest_softmax = logitlens(
                        model, cache[component], tokens_of_interest_bird_rabb, 
                        pos=[t for t in range(cache[component].shape[-2])], 
                        device=device, 
                        softmax=True,
                        save_path=os.path.join(cache_save_dir, f"{component}_logitlens_softmax.npy"),
                        ln_mode="independent",
                    )
                    # Dominant-patch analyses assume llava-1.5's fixed 24x24
                    # image-token grid; anyres models (v1.6/NEXT) raise here and
                    # are skipped -- the paper only uses the 7B examples, and
                    # Appendix C explicitly omits patch maps for other models.
                    try:
                        dominant_patch_map = dominant_patch_map_from_logitlens(
                            logits_of_interest_softmax,
                            tokens_of_interest=tokens_of_interest_bird_rabb,
                            threshold=0.1,
                        )
                        show_dominant_patch_map(
                            dominant_patch_map, image, tokens_of_interest=tokens_of_interest_bird_rabb,
                            x_y_pairs=[
                                [0, 1], # duck vs rabbit
                                ],
                            save_dir=dominant_patch_map_dir,
                        )
                        os.makedirs(os.path.join(dominant_patch_map_dir, prompt_slug), exist_ok=True)
                        zero_ablate_image_embed_with_dominant_patch_map(
                            model, processor, image, text_input, tokens_of_interest_bird_rabb,
                            dominant_patch_map,
                            device=device,
                            save_dir=os.path.join(dominant_patch_map_dir, prompt_slug),
                        )
                    except (ValueError, RuntimeError) as e:
                        print(f"[SKIP] dominant-patch analyses for {model_id}/{image_name}: {e}")

            del cache, curves, image
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        del model, processor
        torch.cuda.empty_cache()


        # mean over all duck-rabbit images
        model_id = args.model
        prompt_slug = "list-every-animal-in-the-image-each-in-one-word-i-see-a"
        # show_dla_from_submodules(
        #     logitlens_cache_dirs,
        #         [0, 1], # bird vs rabbit
        #         ],

        ablation_with_dpm_dir = os.path.join(results_dir, "plots", "ablation_with_dominant_patch_map", model_id, image_name, prompt_slug)
        ablation_with_dpm_cache_dirs = [os.path.join(results_dir, "plots", "dominant_patch_map", model_id, image_name, prompt_slug) for image_name in included_images_by_prompt_slug[prompt_slug]]
        try:
            plot_zero_ablation_effect_with_dominant_patch_map(
                ablation_with_dpm_cache_dirs,
                tokens_of_interest=tokens_of_interest_bird_rabb,
                save_dir=ablation_with_dpm_dir,
            )
        except (FileNotFoundError, ValueError) as e:
            # anyres models skip the dominant-patch stage, so its per-image
            # npy products don't exist; the DLA aggregation below must still run.
            print(f"[SKIP] ablation-with-DPM summary for {model_id}: {e}")
        
        
        dla_agg_root = os.path.join(results_dir, "plots", "layerwise_dla", model_id)
        os.makedirs(dla_agg_root, exist_ok=True)

        for prompt_slug, curves_list in curves_by_prompt_slug.items():
            if len(curves_list) == 0:
                continue

            agg_dir = os.path.join(dla_agg_root, prompt_slug)
            os.makedirs(agg_dir, exist_ok=True)

            meta = prompt_meta_by_slug[prompt_slug]

            with open(os.path.join(agg_dir, "included_images.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_id": model_id,
                        "prompt": meta["prompt"],
                        "prompt_prefix": meta["prompt_prefix"],
                        "prompt_slug": prompt_slug,
                        "n_valid_images": len(included_images_by_prompt_slug[prompt_slug]),
                        "images": included_images_by_prompt_slug[prompt_slug],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            plot_layerwise_dla_mean_std_one_panel(
                curves_by_image=curves_list,
                tokens_of_interest=tokens_of_interest_bird_rabb,
                save_path=os.path.join(agg_dir, "layerwise_dla_mean_std.png"),
                #     f"({model_id})\n"
                #     f"Q:{meta['prompt']} A:{meta['prompt_prefix']}\n"
                #     f"Layerwise DLA at last position over valid VA samples "
                #     f"(n={len(included_images_by_prompt_slug[prompt_slug])})"
            )