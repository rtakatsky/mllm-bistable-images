# query_patching.py

import os
from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json
import math
import argparse
import random
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF


# =========================
# model navigation
# =========================
def get_lm_layers(model) -> List[Any]:
    """
    Expected for LLaVA-style HF models:
      model.language_model.model.layers
    """
    lm = getattr(model, "language_model", model)
    base = getattr(lm, "model", lm)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise AttributeError("Could not find .language_model.model.layers (or .model.layers).")
    return list(layers)


# =========================
# prompt construction
# =========================
def build_text_prompt(
    processor,
    prompt: str,
    prompt_prefix: Optional[str],
    prompt_format: Optional[str],
) -> str:
    """
    Mirror run_vlm_sampling prompt construction.
    """
    if prompt_format is None:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    else:
        text_prompt = prompt_format.format(prompt=prompt)
        if prompt_prefix is not None:
            text_prompt = text_prompt + prompt_prefix
    return text_prompt


# =========================
# centered image helpers
# =========================
def _dataset_item_to_pil_image(sample):
    """
    Robustly extract a PIL image from a dataset item.
    """
    x = sample

    if isinstance(sample, dict):
        if "image" in sample:
            x = sample["image"]
        else:
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
    Build a single centered image using RotatedRGBAImageDataset.
    """
    dataset = RotatedRGBAImageDataset(str(image_path), [center_angle])
    if len(dataset) == 0:
        raise RuntimeError(f"RotatedRGBAImageDataset returned empty dataset for {image_path} @ {center_angle}")
    sample = dataset[0]
    return _dataset_item_to_pil_image(sample)


# =========================
# sequence positions
# =========================
# The sequence layout is read off the expanded input_ids of
# every forward pass instead of the hard-coded llava-1.5 window range(5, 581).
# the FULL image block (base view + AnyRes tiles); "image positions" are the
# base view only (first n_base image tokens; HF prepends the base view), which
# keeps the 24x24 attention maps valid. For llava-1.5 this equals (5, 581, 576).
_IMG_SPAN = None

def _set_image_span_from_inputs(model, inputs, n_base: int = 576):
    global _IMG_SPAN
    ids = inputs["input_ids"][0].tolist()
    img_id = int(model.config.image_token_index)
    pos = [i for i, t in enumerate(ids) if t == img_id]
    if len(pos) < n_base:
        raise ValueError(f"only {len(pos)} image tokens in input_ids (< {n_base})")
    _IMG_SPAN = (pos[0], pos[-1] + 1, n_base)
    return _IMG_SPAN


def positions_from_mode(
    seq_len: int,
    mode: str,
    *,
    pre_img_tokens: int = 5,
    image_token_count: int = 576,
    query_end_pos: Optional[int] = None,
) -> List[int]:
    """
    Sequence layout (teacher-forced interpretation):
      [pre-text tokens][image patch tokens][query tokens][gen(prefix) tokens]

    Modes:
      - query:     tokens after image patches and before prompt_prefix
      - gen:       prompt_prefix tokens (includes the last token)
      - query_gen: all tokens after image patches
      - last:      last token only
    """
    if seq_len <= 0:
        return []

    if _IMG_SPAN is not None:
        img_end = min(_IMG_SPAN[1], seq_len)          # after the FULL image block
    else:
        img_end = min(pre_img_tokens + image_token_count, seq_len)

    if query_end_pos is None:
        query_end_pos = seq_len
    query_end_pos = max(img_end, min(int(query_end_pos), seq_len))

    query_pos = list(range(img_end, query_end_pos))
    gen_pos = list(range(query_end_pos, seq_len))
    query_gen_pos = list(range(img_end, seq_len))

    if mode == "last":
        return [seq_len - 1]
    if mode == "query":
        return query_pos
    if mode == "gen":
        return gen_pos
    if mode == "query_gen":
        return query_gen_pos

    raise ValueError(f"Unknown mode: {mode}")


def image_positions_from_layout(
    seq_len: int,
    *,
    pre_img_tokens: int = 5,
    image_token_count: int = 576,
) -> List[int]:
    if _IMG_SPAN is not None:
        img_start = min(_IMG_SPAN[0], seq_len)
        img_end = min(_IMG_SPAN[0] + _IMG_SPAN[2], seq_len)   # base view only
    else:
        img_start = min(pre_img_tokens, seq_len)
        img_end = min(pre_img_tokens + image_token_count, seq_len)
    return list(range(img_start, img_end))


# =========================
# layer range helpers
# =========================
def normalize_layer_ranges(
    ranges: List[Tuple[int, int]],
    n_layers: int,
) -> Tuple[List[List[int]], List[str]]:
    """
    Inclusive ranges -> explicit lists, clamped to valid layer indices.
    """
    out_lists = []
    out_labels = []
    for a, b in ranges:
        lo = max(0, min(int(a), int(b)))
        hi = min(n_layers - 1, max(int(a), int(b)))
        if lo > hi:
            out_lists.append([])
            out_labels.append(f"{a}-{b}(empty)")
        else:
            ls = list(range(lo, hi + 1))
            out_lists.append(ls)
            out_labels.append(f"{ls[0]}-{ls[-1]}")
    return out_lists, out_labels


# =========================
# forward + logit diff
# =========================
def prepare_inputs(processor, image, text_prompt: str, device) -> Dict[str, torch.Tensor]:
    return processor(images=[image], text=[text_prompt], return_tensors="pt").to(device)


@torch.no_grad()
def next_token_logitdiff_from_inputs(
    model,
    inputs: Dict[str, torch.Tensor],
    tid_a: int,
    tid_b: int,
) -> float:
    """
    Logit diff for the first generated token after the prefix.
    """
    out = model(**inputs, use_cache=False)
    logits_last = out.logits[0, -1, :]
    return float((logits_last[tid_a] - logits_last[tid_b]).detach().cpu())


# =========================
# attention extraction helpers
# =========================
def _extract_image_attention_from_attentions(
    attentions,
    *,
    mode: str,
    pre_img_tokens: int,
    image_token_count: int,
    query_end_pos: Optional[int],
):
    """
    Extract raw image attention and image-only ratios:
      attn_img[layer]   = [H, Pq, Nimg]
      attn_ratio[layer] = normalized over image keys only, [H, Pq, Nimg]
    """
    if attentions is None or len(attentions) == 0:
        raise RuntimeError("No attentions returned.")

    seq_len = int(attentions[0].shape[-1])
    query_positions = positions_from_mode(
        seq_len,
        mode,
        pre_img_tokens=pre_img_tokens,
        image_token_count=image_token_count,
        query_end_pos=query_end_pos,
    )
    image_positions = image_positions_from_layout(
        seq_len,
        pre_img_tokens=pre_img_tokens,
        image_token_count=image_token_count,
    )

    attn_img: Dict[int, torch.Tensor] = {}
    attn_ratio: Dict[int, torch.Tensor] = {}

    if len(query_positions) == 0 or len(image_positions) == 0:
        for li, A in enumerate(attentions):
            H = int(A.shape[1])
            empty = A.new_empty((H, 0, 0)).detach()
            attn_img[li] = empty
            attn_ratio[li] = empty
        return attn_img, attn_ratio, query_positions, image_positions

    q_t = torch.tensor(query_positions, dtype=torch.long, device=attentions[0].device)
    k_t = torch.tensor(image_positions, dtype=torch.long, device=attentions[0].device)

    for li, A in enumerate(attentions):
        # A: [B,H,Q,K] -> [H,Pq,Nimg]
        img = A[0].index_select(1, q_t).index_select(2, k_t).detach()
        denom = img.sum(dim=-1, keepdim=True)
        ratio = torch.where(
            denom > 1e-12,
            img / denom.clamp_min(1e-12),
            torch.zeros_like(img),
        )
        attn_img[li] = img
        attn_ratio[li] = ratio

    return attn_img, attn_ratio, query_positions, image_positions


@torch.no_grad()
def run_and_capture_image_attention_mode(
    model,
    inputs: Dict[str, torch.Tensor],
    tid_a: int,
    tid_b: int,
    *,
    mode: str,
    pre_img_tokens: int,
    image_token_count: int,
    query_end_pos: Optional[int],
):
    """
    Run and capture:
      - logitdiff
      - image attention slice
      - image attention ratio (normalized within image keys)
    """
    _set_image_span_from_inputs(model, inputs)
    out = model(**inputs, use_cache=False, output_attentions=True)
    logits_last = out.logits[0, -1, :]
    ld = float((logits_last[tid_a] - logits_last[tid_b]).detach().cpu())

    attn_img, attn_ratio, query_positions, image_positions = _extract_image_attention_from_attentions(
        out.attentions,
        mode=mode,
        pre_img_tokens=pre_img_tokens,
        image_token_count=image_token_count,
        query_end_pos=query_end_pos,
    )

    return ld, attn_img, attn_ratio, query_positions, image_positions


# =========================
# Q capture and hybrid attention
# =========================
@torch.no_grad()
def run_and_capture_q_and_image_attention_mode(
    model,
    inputs: Dict[str, torch.Tensor],
    tid_a: int,
    tid_b: int,
    *,
    mode: str,
    pre_img_tokens: int,
    image_token_count: int,
    query_end_pos: Optional[int],
):
    """
    Run once and capture:
      - logitdiff
      - q_proj outputs at selected positions
      - post-softmax image attention
      - image-only attention ratios
    """
    _set_image_span_from_inputs(model, inputs)
    layers = get_lm_layers(model)
    q_cache: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_q_hook(li: int):
        def hook(module, inp, out):
            seq_len = int(out.shape[1])
            positions = positions_from_mode(
                seq_len,
                mode,
                pre_img_tokens=pre_img_tokens,
                image_token_count=image_token_count,
                query_end_pos=query_end_pos,
            )
            if len(positions) == 0:
                q_cache[li] = out.new_empty((0, out.shape[-1])).detach()
            else:
                pos_t = torch.tensor(positions, dtype=torch.long, device=out.device)
                q_cache[li] = out[0].index_select(0, pos_t).detach()
            return out
        return hook

    for li, layer in enumerate(layers):
        q_proj = getattr(layer.self_attn, "q_proj", None)
        if q_proj is None:
            raise AttributeError(f"Layer {li}: self_attn.q_proj not found.")
        hooks.append(q_proj.register_forward_hook(make_q_hook(li)))

    try:
        out = model(**inputs, use_cache=False, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    logits_last = out.logits[0, -1, :]
    ld = float((logits_last[tid_a] - logits_last[tid_b]).detach().cpu())

    attn_img, attn_ratio, query_positions, image_positions = _extract_image_attention_from_attentions(
        out.attentions,
        mode=mode,
        pre_img_tokens=pre_img_tokens,
        image_token_count=image_token_count,
        query_end_pos=query_end_pos,
    )

    return ld, q_cache, attn_img, attn_ratio, query_positions, image_positions


@torch.no_grad()
def run_with_q_patch_and_capture_image_attention_mode(
    model,
    inputs: Dict[str, torch.Tensor],
    *,
    q_src: Dict[int, torch.Tensor],
    layers_to_patch: List[int],
    mode: str,
    pre_img_tokens: int,
    image_token_count: int,
    query_end_pos: Optional[int],
):
    """
    Patch q_proj outputs from q_src into the target run at selected positions,
    then return the resulting image attention slice and its within-image ratio.
    """
    _set_image_span_from_inputs(model, inputs)
    if len(layers_to_patch) == 0:
        out = model(**inputs, use_cache=False, output_attentions=True)
        return _extract_image_attention_from_attentions(
            out.attentions,
            mode=mode,
            pre_img_tokens=pre_img_tokens,
            image_token_count=image_token_count,
            query_end_pos=query_end_pos,
        )

    layers = get_lm_layers(model)
    layers_set = set(layers_to_patch)
    hooks = []

    def make_q_patch_hook(li: int):
        def hook(module, inp, out):
            if li not in layers_set or li not in q_src:
                return out

            seq_len = int(out.shape[1])
            positions = positions_from_mode(
                seq_len,
                mode,
                pre_img_tokens=pre_img_tokens,
                image_token_count=image_token_count,
                query_end_pos=query_end_pos,
            )
            if len(positions) == 0 or q_src[li].numel() == 0:
                return out

            pos_t = torch.tensor(positions, dtype=torch.long, device=out.device)
            P = min(len(positions), q_src[li].shape[0])
            if P == 0:
                return out

            patched = out.clone()
            patched[0].index_copy_(0, pos_t[:P], q_src[li][:P].to(out.device, dtype=out.dtype))
            return patched
        return hook

    try:
        for li, layer in enumerate(layers):
            q_proj = getattr(layer.self_attn, "q_proj", None)
            if q_proj is None:
                raise AttributeError(f"Layer {li}: self_attn.q_proj not found.")
            hooks.append(q_proj.register_forward_hook(make_q_patch_hook(li)))

        out = model(**inputs, use_cache=False, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    return _extract_image_attention_from_attentions(
        out.attentions,
        mode=mode,
        pre_img_tokens=pre_img_tokens,
        image_token_count=image_token_count,
        query_end_pos=query_end_pos,
    )


# =========================
# reverse patch image-attention ratios onto target run
# =========================
@torch.no_grad()
def run_with_image_attention_ratio_patch_mode(
    model,
    inputs: Dict[str, torch.Tensor],
    tid_a: int,
    tid_b: int,
    *,
    attn_ratio_src: Dict[int, torch.Tensor],
    layers_to_patch: List[int],
    mode: str,
    pre_img_tokens: int,
    image_token_count: int,
    query_end_pos: Optional[int],
):
    """
    Reverse-patch:
      - target run stays modulated
      - replace only the within-image attention distribution
        using source ratios
      - preserve the target run's total image-attention mass per row
    """
    _set_image_span_from_inputs(model, inputs)
    if len(layers_to_patch) == 0:
        return next_token_logitdiff_from_inputs(model, inputs, tid_a, tid_b)

    layers = get_lm_layers(model)
    layers_set = set(layers_to_patch)
    hooks = []
    vproj_buf: Dict[int, torch.Tensor] = {}

    seq_len = int(inputs["input_ids"].shape[1])
    query_positions = positions_from_mode(
        seq_len,
        mode,
        pre_img_tokens=pre_img_tokens,
        image_token_count=image_token_count,
        query_end_pos=query_end_pos,
    )
    image_positions = image_positions_from_layout(
        seq_len,
        pre_img_tokens=pre_img_tokens,
        image_token_count=image_token_count,
    )

    if len(query_positions) == 0 or len(image_positions) == 0:
        return next_token_logitdiff_from_inputs(model, inputs, tid_a, tid_b)

    q_t = torch.tensor(query_positions, dtype=torch.long, device=inputs["input_ids"].device)
    k_t = torch.tensor(image_positions, dtype=torch.long, device=inputs["input_ids"].device)

    def make_vproj_hook(li: int):
        def hook(module, inp, out):
            vproj_buf[li] = out.detach()
        return hook

    def make_attn_rewrite_hook(li: int):
        def hook(module, inp, out):
            if li not in layers_set or li not in attn_ratio_src:
                return out

            if not isinstance(out, tuple) or len(out) < 2:
                return out

            attn_weights = out[1]
            rest = out[2:] if len(out) > 2 else ()

            if attn_weights is None:
                raise RuntimeError(
                    "Attention weights are None. Need output_attentions=True and a backend that returns them."
                )
            if li not in vproj_buf:
                raise RuntimeError(f"Missing v_proj output for layer {li}.")

            v = vproj_buf[li]   # [B, kv, D]
            B, H, q_len, kv_len = attn_weights.shape
            if B != 1:
                raise RuntimeError(f"Expected batch size 1, got {B}")

            hd = v.shape[-1] // H
            value_states = v.view(B, kv_len, H, hd).transpose(1, 2).contiguous()  # [B,H,K,hd]

            w = attn_weights.clone()   # [B,H,Q,K]
            src_ratio = attn_ratio_src[li].to(w.device, dtype=w.dtype)   # [H,P,N]

            P = min(src_ratio.shape[1], len(query_positions))
            N = min(src_ratio.shape[2], len(image_positions))
            if P > 0 and N > 0:
                current_img = w[0, :, q_t[:P, None], k_t[None, :N]]   # [H,P,N]
                target_mass = current_img.sum(dim=-1, keepdim=True)    # [H,P,1]

                src_ratio_local = src_ratio[:, :P, :N]
                denom = src_ratio_local.sum(dim=-1, keepdim=True)
                src_ratio_local = torch.where(
                    denom > 1e-12,
                    src_ratio_local / denom.clamp_min(1e-12),
                    torch.zeros_like(src_ratio_local),
                )

                patched_img = target_mass * src_ratio_local
                w[0, :, q_t[:P, None], k_t[None, :N]] = patched_img

            out_heads = torch.matmul(w, value_states)  # [B,H,Q,hd]
            out_concat = out_heads.transpose(1, 2).contiguous().view(B, q_len, H * hd)
            new_attn_output = module.o_proj(out_concat)

            if len(out) == 2:
                return (new_attn_output, w)
            return (new_attn_output, w, *rest)

        return hook

    try:
        for li, layer in enumerate(layers):
            if li in layers_set:
                hooks.append(layer.self_attn.v_proj.register_forward_hook(make_vproj_hook(li)))
                hooks.append(layer.self_attn.register_forward_hook(make_attn_rewrite_hook(li)))

        out = model(**inputs, use_cache=False, output_attentions=True)
        logits_last = out.logits[0, -1, :]
        ld = float((logits_last[tid_a] - logits_last[tid_b]).detach().cpu())
        return ld

    finally:
        for h in hooks:
            h.remove()


# =========================
# map aggregation
# =========================
def mean_ratio_diff_map_over_setup(
    ratio_a: Dict[int, torch.Tensor],
    ratio_b: Dict[int, torch.Tensor],
    layers_to_patch: List[int],
) -> Optional[np.ndarray]:
    """
    Returns mean image-token difference map [Nimg] averaged over:
      selected layers, heads, query positions
    """
    maps = []

    for li in layers_to_patch:
        if li not in ratio_a or li not in ratio_b:
            continue

        A = ratio_a[li]
        B = ratio_b[li]
        if A.ndim != 3 or B.ndim != 3:
            continue
        if A.numel() == 0 or B.numel() == 0:
            continue

        H = min(A.shape[0], B.shape[0])
        P = min(A.shape[1], B.shape[1])
        N = min(A.shape[2], B.shape[2])
        if H == 0 or P == 0 or N == 0:
            continue

        diff = (A[:H, :P, :N] - B[:H, :P, :N]).mean(dim=(0, 1))  # [N]
        maps.append(diff.detach().cpu().numpy().astype(np.float32))

    if len(maps) == 0:
        return None

    return np.mean(np.stack(maps, axis=0), axis=0)


def plot_mean_attention_change_map(
    maps: List[np.ndarray],
    *,
    title: str,
    save_path: str,
    image_token_count: int,
):
    if len(maps) == 0:
        print(f"[WARN] No attention-change maps to plot: {save_path}")
        return

    arr = np.stack(maps, axis=0)        # [M, Nimg]
    mean_map = arr.mean(axis=0)         # [Nimg]

    grid = int(round(math.sqrt(image_token_count)))
    if grid * grid != image_token_count:
        raise ValueError(f"image_token_count={image_token_count} is not a square.")

    mean_map_2d = mean_map.reshape(grid, grid)
    vmax = float(np.max(np.abs(mean_map_2d)) + 1e-9)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(mean_map_2d, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("mean image-attention ratio change")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    np.save(os.path.splitext(save_path)[0] + ".npy", mean_map_2d)


# =========================
# fit helpers
# =========================
def fit_through_origin(records: List[Dict[str, Any]]) -> Dict[str, float]:
    xs = np.array([r["x"] for r in records if np.isfinite(r["x"]) and np.isfinite(r["y"])], dtype=float)
    ys = np.array([r["y"] for r in records if np.isfinite(r["x"]) and np.isfinite(r["y"])], dtype=float)

    if xs.size == 0:
        return {"n": 0, "slope": float("nan"), "r2_origin": float("nan")}

    denom = float(np.sum(xs ** 2))
    if abs(denom) < 1e-12:
        return {"n": int(xs.size), "slope": float("nan"), "r2_origin": float("nan")}

    slope = float(np.sum(xs * ys) / denom)
    yhat = slope * xs
    sse = float(np.sum((ys - yhat) ** 2))
    sst0 = float(np.sum(ys ** 2))
    r2_origin = float("nan") if sst0 < 1e-12 else float(1.0 - sse / sst0)

    return {"n": int(xs.size), "slope": slope, "r2_origin": r2_origin}


# =========================
# plotting
# =========================
def plot_effect_scatter(
    records: List[Dict[str, Any]],
    *,
    title: str,
    save_path: str,
    ylabel: str,
):
    """
    Generic scatter:
      x = modulation effect
      y = effect explained/recovered
    """
    if not records:
        print(f"[WARN] No records to plot: {save_path}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    rot = [r for r in records if r["condition"] == "rotation" and np.isfinite(r["x"]) and np.isfinite(r["y"])]
    red = [r for r in records if r["condition"] == "red_circle" and np.isfinite(r["x"]) and np.isfinite(r["y"])]
    td = [r for r in records if r["condition"] == "top_down" and np.isfinite(r["x"]) and np.isfinite(r["y"])]

    if rot:
        ax.scatter(
            [r["x"] for r in rot],
            [r["y"] for r in rot],
            marker="x",
            s=45,
            linewidths=1.3,
            alpha=0.85,
            label="rotation",
            zorder=2,
        )

    if red:
        ax.scatter(
            [r["x"] for r in red],
            [r["y"] for r in red],
            marker="o",
            facecolors="none",
            edgecolors="tab:orange",
            s=55,
            linewidths=1.4,
            alpha=0.95,
            label="red_circle",
            zorder=3,
        )

    if td:
        ax.scatter(
            [r["x"] for r in td],
            [r["y"] for r in td],
            marker="s",
            s=42,
            alpha=0.85,
            label="top_down",
            zorder=2,
        )

    ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axvline(0.0, linestyle=":", linewidth=1.0, alpha=0.6)

    xs = np.array([r["x"] for r in records if np.isfinite(r["x"])], dtype=float)
    ys = np.array([r["y"] for r in records if np.isfinite(r["y"])], dtype=float)
    if len(xs) and len(ys):
        lim = float(max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1e-6))
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_ylim(-lim * 1.05, lim * 1.05)

    ax.set_xlabel("Modulation effect: (logitdiff_mod - logitdiff_base)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_combined_effect_scatter_with_fits(
    bottom_up_records: List[Dict[str, Any]],
    top_down_records: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Combined scatter with:
      - bottom-up induced (rotation + red_circle)
      - top-down induced (x->r/d)
    and two through-origin fitted lines.
    """
    combined = bottom_up_records + top_down_records
    if not combined:
        print(f"[WARN] No combined records to plot: {save_path}")
        return {
            "bottom_up": {"n": 0, "slope": float("nan"), "r2_origin": float("nan")},
            "top_down": {"n": 0, "slope": float("nan"), "r2_origin": float("nan")},
        }

    fig, ax = plt.subplots(figsize=(7.5, 6.2))

    rot = [r for r in bottom_up_records if r["condition"] == "rotation" and np.isfinite(r["x"]) and np.isfinite(r["y"])]
    red = [r for r in bottom_up_records if r["condition"] == "red_circle" and np.isfinite(r["x"]) and np.isfinite(r["y"])]
    td = [r for r in top_down_records if r["condition"] == "top_down" and np.isfinite(r["x"]) and np.isfinite(r["y"])]

    if rot:
        ax.scatter(
            [r["x"] for r in rot],
            [r["y"] for r in rot],
            marker="x",
            s=45,
            linewidths=1.3,
            alpha=0.80,
            label="bottom-up: rotation",
            zorder=2,
            color="black",
        )

    if red:
        ax.scatter(
            [r["x"] for r in red],
            [r["y"] for r in red],
            marker=".",
            color="black",
            s=55,
            alpha=0.95,
            label="bottom-up: red_circle",
            zorder=3,
        )

    if td:
        ax.scatter(
            [r["x"] for r in td],
            [r["y"] for r in td],
            marker="s",
            s=42,
            alpha=0.85,
            label="top-down: x→r/d",
            zorder=2,
            color="tab:green",
        )

    ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axvline(0.0, linestyle=":", linewidth=1.0, alpha=0.6)

    xs = np.array([r["x"] for r in combined if np.isfinite(r["x"])], dtype=float)
    ys = np.array([r["y"] for r in combined if np.isfinite(r["y"])], dtype=float)
    lim = 1.0
    if len(xs) and len(ys):
        lim = float(max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1e-6))
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_ylim(-lim * 1.05, lim * 1.05)

    fit_bottom = fit_through_origin(bottom_up_records)
    fit_top = fit_through_origin(top_down_records)

    xx = np.array([-lim * 1.05, lim * 1.05], dtype=float)

    if np.isfinite(fit_bottom["slope"]):
        ax.plot(
            xx,
            fit_bottom["slope"] * xx,
            linewidth=2.2,
            color="black",
            linestyle="-",
            label=f"bottom-up fit (β={fit_bottom['slope']:.3f})",
            zorder=4,
        )

    if np.isfinite(fit_top["slope"]):
        ax.plot(
            xx,
            fit_top["slope"] * xx,
            linewidth=2.2,
            color="darkgreen",
            linestyle="--",
            label=f"top-down fit (β={fit_top['slope']:.3f})",
            zorder=4,
        )

    ax.set_xlabel("Total Modulation Effect: Δ(Duck - Rabbit)")
    ax.set_ylabel("Effect explained by query-side changes in text→image attention")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return {"bottom_up": fit_bottom, "top_down": fit_top}


# =========================
# main
# =========================
if __name__ == "__main__":
    from utils import *  # load_vlm, slugify, RotatedRGBAImageDataset, RedCircleImageDataset, get_boundary

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--base_dir", type=str, default=PROJECT_ROOT)
    parser.add_argument("--rotation_offset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--radius_ratio", type=float, default=1 / 14)
    parser.add_argument("--stride_ratio", type=float, default=1 / 7)
    parser.add_argument("--line_width", type=int, default=4)
    parser.add_argument("--pre_img_tokens", type=int, default=5)
    parser.add_argument("--image_token_count", type=int, default=(336 // 14) ** 2)  # 576
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading/inference; replot the combined scatters from tables/<model>/combined_bottomup_vs_topdown/*.json (CPU-friendly).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_id = args.model
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    base_dir = args.base_dir
    duck_rabbit_dir = os.path.join(base_dir, "data", "duck_rabbit")
    boundaries_path = os.path.join(duck_rabbit_dir, "boundaries.json")

    os.environ["HF_HOME"] = os.path.join(base_dir, "huggingface_cache")

    results_dir = os.path.join(base_dir, "outputs", "query_patching")
    scatter_root = os.path.join(results_dir, "plots", model_id, "scatter")
    table_root = os.path.join(results_dir, "tables", model_id)
    maps_root = os.path.join(results_dir, "plots", model_id, "mean_attention_change_maps")
    centered_cache_dir = os.path.join(results_dir, "_centered_cache", model_id)
    os.makedirs(scatter_root, exist_ok=True)
    os.makedirs(table_root, exist_ok=True)
    os.makedirs(maps_root, exist_ok=True)
    os.makedirs(centered_cache_dir, exist_ok=True)

    with open(boundaries_path, "r", encoding="utf-8") as f:
        boundaries = json.load(f)

    vlm_dict = {
        "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
        "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
        "llava-v1.6-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava-v1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llama3-llava-next-8b": "llava-hf/llama3-llava-next-8b-hf",
    }

    duck_rabbit_tokens = {
        "llava-1.5-7b": ("▁du", "▁rabb"),
        "llava-1.5-13b": ("▁du", "▁rabb"),
        "llava-v1.6-vicuna-7b": ("▁du", "▁rabb"),
        "llava-v1.6-mistral-7b": ("▁duck", "▁rab"),
        "llama3-llava-next-8b": ("Ġduck", "Ġrabbit"),
    }

    bottom_up_prompts = [
        ["List every animal in the image, each in one word.", "There's a"],
        ["List every animal in the image, each in one word.", "I see a"],
    ]

    top_down_prompts = {
        "x": ["List every animal in the image, each in one word. It starts with 'x'.", "I see a"],
        "r": ["List every animal in the image, each in one word. It starts with 'r'.", "I see a"],
        "d": ["List every animal in the image, each in one word. It starts with 'd'.", "I see a"],
    }

    top_down_boundary_slug = slugify("List every animal in the image, each in one word." + "I see a")
    bottom_up_prompt_slug_for_combined = slugify("List every animal in the image, each in one word." + "I see a")

    layer_ranges = [(0, 19), (20, 26), (30, 31), (0, 32)]
    patch_modes = ["query_gen", "query", "gen", "last"]

    if args.plot_only:
        # Replot the paper's combined bottom-up vs top-down scatters from the
        # saved record tables; no model needed.
        import glob as _glob
        combined_scatter_dir = os.path.join(scatter_root, "combined_bottomup_vs_topdown")
        combined_table_dir = os.path.join(table_root, "combined_bottomup_vs_topdown")
        os.makedirs(combined_scatter_dir, exist_ok=True)
        table_files = sorted(p for p in _glob.glob(os.path.join(combined_table_dir, "*__layers_*.json"))
                             if not p.endswith("__fit.json"))
        if not table_files:
            raise SystemExit(f"[PLOT-ONLY] no combined tables under {combined_table_dir}")
        for jp in table_files:
            with open(jp, "r", encoding="utf-8") as f:
                recs = json.load(f)
            recs_bottom = [r for r in recs if r.get("condition") in ("rotation", "red_circle")]
            recs_top = [r for r in recs if r.get("condition") == "top_down"]
            stem = os.path.splitext(os.path.basename(jp))[0]
            plot_combined_effect_scatter_with_fits(
                recs_bottom, recs_top,
                save_path=os.path.join(combined_scatter_dir, f"{stem}.png"),
            )
            print(f"[PLOT-ONLY] replotted {stem} ({len(recs_bottom)} bottom-up, {len(recs_top)} top-down records)")
        raise SystemExit(0)

    print(f"Loading model {model_id}...")
    processor, model, prompt_format = load_vlm(vlm_dict[model_id], return_prompt_format=True)
    model.to(device).eval()

    layers = get_lm_layers(model)
    n_layers = len(layers)
    layer_lists, layer_labels = normalize_layer_ranges(layer_ranges, n_layers)

    tok_a_str, tok_b_str = duck_rabbit_tokens[model_id]
    tid_a = processor.tokenizer.convert_tokens_to_ids(tok_a_str)
    tid_b = processor.tokenizer.convert_tokens_to_ids(tok_b_str)

    print("[INFO] For first-token evaluation with teacher-forced prompt_prefix:")
    print("       mode='query'     = tokens after image patches and before prompt_prefix")
    print("       mode='gen'       = prompt_prefix tokens (includes the last token)")
    print("       mode='query_gen' = all tokens after image patches")
    print("       mode='last'      = last token only")

    # Expanded pipeline-generated set, deterministic order (va_set_manifest.json);
    # partial runs therefore cover a well-defined "first N" subset.
    with open(os.path.join(DATA_DIR, "duck_rabbit", "va_set_manifest.json")) as f:
        va_names = [f"va_s{seed:03d}" for seed in json.load(f)["seeds"]]

    # keep bottom-up records for neutral I see a to combine with top-down
    bottom_up_records_for_combined = None

    # -------------------------
    # A) bottom-up INDIRECT image-attention effect
    # -------------------------
    for prompt, prompt_prefix in bottom_up_prompts:
        prompt_slug = slugify(prompt + prompt_prefix)
        text_prompt = build_text_prompt(processor, prompt, prompt_prefix, prompt_format)
        text_prompt_no_prefix = build_text_prompt(processor, prompt, None, prompt_format)

        prompt_scatter_dir = os.path.join(scatter_root, prompt_slug)
        prompt_table_dir = os.path.join(table_root, prompt_slug)
        prompt_maps_dir = os.path.join(maps_root, "bottom_up_indirect", prompt_slug)
        os.makedirs(prompt_scatter_dir, exist_ok=True)
        os.makedirs(prompt_table_dir, exist_ok=True)
        os.makedirs(prompt_maps_dir, exist_ok=True)

        valid_va = []
        for va in va_names:
            b = get_boundary(boundaries, model_id, va, prompt_slug)
            if b is not None:
                valid_va.append(va)

        print(f"\n[INFO] {model_id} | {prompt_slug} | valid_va = {valid_va}")

        records_by_mode_and_range: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            mode: {lab: [] for lab in layer_labels} for mode in patch_modes
        }
        mean_maps_by_mode_and_range: Dict[str, Dict[str, List[np.ndarray]]] = {
            mode: {lab: [] for lab in layer_labels} for mode in patch_modes
        }

        for va in tqdm(valid_va, desc=prompt_slug):
            center_angle = get_boundary(boundaries, model_id, va, prompt_slug)
            if center_angle is None:
                continue

            orig_path = os.path.join(duck_rabbit_dir, f"{va}.png")
            if not os.path.exists(orig_path):
                print(f"[WARN] Missing original image: {orig_path}")
                continue

            # rotation base image at center angle
            base_rot_dataset = RotatedRGBAImageDataset(orig_path, [int(center_angle)])
            img_base_rot = _dataset_item_to_pil_image(base_rot_dataset[0])
            inputs_base_rot = prepare_inputs(processor, img_base_rot, text_prompt, device)
            inputs_base_rot_no_prefix = prepare_inputs(processor, img_base_rot, text_prompt_no_prefix, device)
            query_end_pos = int(inputs_base_rot_no_prefix["input_ids"].shape[1])

            ld_base_rot = next_token_logitdiff_from_inputs(model, inputs_base_rot, tid_a, tid_b)

            # centered base image for red-circle condition
            img_base_red = make_centered_analysis_image(orig_path, float(center_angle))
            inputs_base_red = prepare_inputs(processor, img_base_red, text_prompt, device)
            ld_base_red = next_token_logitdiff_from_inputs(model, inputs_base_red, tid_a, tid_b)

            centered_path = os.path.join(centered_cache_dir, f"{va}__center_{int(round(center_angle))}.png")
            if not os.path.exists(centered_path):
                img_base_red.save(centered_path)

            # precompute base attention ratios for each mode
            base_rot_ratio_by_mode = {}
            base_red_ratio_by_mode = {}
            for mode in patch_modes:
                _, _, ratio_rot, _, _ = run_and_capture_image_attention_mode(
                    model,
                    inputs_base_rot,
                    tid_a,
                    tid_b,
                    mode=mode,
                    pre_img_tokens=args.pre_img_tokens,
                    image_token_count=args.image_token_count,
                    query_end_pos=query_end_pos,
                )
                base_rot_ratio_by_mode[mode] = ratio_rot

                _, _, ratio_red, _, _ = run_and_capture_image_attention_mode(
                    model,
                    inputs_base_red,
                    tid_a,
                    tid_b,
                    mode=mode,
                    pre_img_tokens=args.pre_img_tokens,
                    image_token_count=args.image_token_count,
                    query_end_pos=query_end_pos,
                )
                base_red_ratio_by_mode[mode] = ratio_red

            # A1) rotation modulations
            deltas = list(range(-args.rotation_offset, args.rotation_offset + 1))
            rot_angles = [int(center_angle) + d for d in deltas]
            rot_dataset = RotatedRGBAImageDataset(orig_path, rot_angles)

            for d, sample_mod in zip(deltas, [rot_dataset[i] for i in range(len(rot_dataset))]):
                img_mod = _dataset_item_to_pil_image(sample_mod)
                inputs_mod = prepare_inputs(processor, img_mod, text_prompt, device)

                for mode in patch_modes:
                    ld_mod, q_mod, _, ratio_mod, positions, image_positions = run_and_capture_q_and_image_attention_mode(
                        model,
                        inputs_mod,
                        tid_a,
                        tid_b,
                        mode=mode,
                        pre_img_tokens=args.pre_img_tokens,
                        image_token_count=args.image_token_count,
                        query_end_pos=query_end_pos,
                    )
                    x = ld_mod - ld_base_rot

                    base_ratio = base_rot_ratio_by_mode[mode]

                    for lab, layer_idxs in zip(layer_labels, layer_lists):
                        # remove all image-attention change by patching base ratios onto mod run
                        ld_mod_to_base_ratio = run_with_image_attention_ratio_patch_mode(
                            model,
                            inputs_mod,
                            tid_a,
                            tid_b,
                            attn_ratio_src=base_ratio,
                            layers_to_patch=layer_idxs,
                            mode=mode,
                            pre_img_tokens=args.pre_img_tokens,
                            image_token_count=args.image_token_count,
                            query_end_pos=query_end_pos,
                        )

                        # compute hybrid query-side image-attention ratios on the BASE image
                        _, hybrid_ratio, _, _ = run_with_q_patch_and_capture_image_attention_mode(
                            model,
                            inputs_base_rot,
                            q_src=q_mod,
                            layers_to_patch=layer_idxs,
                            mode=mode,
                            pre_img_tokens=args.pre_img_tokens,
                            image_token_count=args.image_token_count,
                            query_end_pos=query_end_pos,
                        )

                        # keep only the indirect/query-side image-attention change
                        ld_mod_to_hybrid_ratio = run_with_image_attention_ratio_patch_mode(
                            model,
                            inputs_mod,
                            tid_a,
                            tid_b,
                            attn_ratio_src=hybrid_ratio,
                            layers_to_patch=layer_idxs,
                            mode=mode,
                            pre_img_tokens=args.pre_img_tokens,
                            image_token_count=args.image_token_count,
                            query_end_pos=query_end_pos,
                        )

                        # indirect effect = effect added back by hybrid over base-ratio patch
                        y = ld_mod_to_hybrid_ratio - ld_mod_to_base_ratio

                        rec = {
                            "condition": "rotation",
                            "family": "bottom_up",
                            "va": va,
                            "delta": int(d),
                            "mode": mode,
                            "layers": lab,
                            "n_positions": len(positions),
                            "n_image_positions": len(image_positions),
                            "x": float(x),
                            "y": float(y),
                            "ld_base": float(ld_base_rot),
                            "ld_mod": float(ld_mod),
                            "ld_mod_to_base_ratio": float(ld_mod_to_base_ratio),
                            "ld_mod_to_hybrid_ratio": float(ld_mod_to_hybrid_ratio),
                            "removed_full_image_attention": float(ld_mod - ld_mod_to_base_ratio),
                            "indirect_image_attention_effect": float(y),
                        }
                        records_by_mode_and_range[mode][lab].append(rec)

                        diff_map = mean_ratio_diff_map_over_setup(
                            hybrid_ratio,
                            base_ratio,
                            layer_idxs,
                        )
                        if diff_map is not None:
                            mean_maps_by_mode_and_range[mode][lab].append(diff_map)

            # A2) red circle modulations
            red_dataset = RedCircleImageDataset(
                centered_path,
                radius_ratio=args.radius_ratio,
                stride_ratio=args.stride_ratio,
                line_width=args.line_width,
            )

            if len(red_dataset) == 0:
                print(f"[WARN] RedCircleImageDataset is empty for {va}")
            else:
                k_red = min(len(deltas), len(red_dataset))
                red_indices = random.sample(range(len(red_dataset)), k_red)

                for idx in red_indices:
                    img_red = _dataset_item_to_pil_image(red_dataset[idx])
                    inputs_mod_red = prepare_inputs(processor, img_red, text_prompt, device)

                    for mode in patch_modes:
                        ld_mod, q_mod, _, ratio_mod, positions, image_positions = run_and_capture_q_and_image_attention_mode(
                            model,
                            inputs_mod_red,
                            tid_a,
                            tid_b,
                            mode=mode,
                            pre_img_tokens=args.pre_img_tokens,
                            image_token_count=args.image_token_count,
                            query_end_pos=query_end_pos,
                        )
                        x = ld_mod - ld_base_red

                        base_ratio = base_red_ratio_by_mode[mode]

                        ld_mod_to_base_ratio = run_with_image_attention_ratio_patch_mode(
                            model,
                            inputs_mod_red,
                            tid_a,
                            tid_b,
                            attn_ratio_src=base_ratio,
                            layers_to_patch=[],
                            mode=mode,
                            pre_img_tokens=args.pre_img_tokens,
                            image_token_count=args.image_token_count,
                            query_end_pos=query_end_pos,
                        ) if False else None

                        for lab, layer_idxs in zip(layer_labels, layer_lists):
                            ld_mod_to_base_ratio = run_with_image_attention_ratio_patch_mode(
                                model,
                                inputs_mod_red,
                                tid_a,
                                tid_b,
                                attn_ratio_src=base_ratio,
                                layers_to_patch=layer_idxs,
                                mode=mode,
                                pre_img_tokens=args.pre_img_tokens,
                                image_token_count=args.image_token_count,
                                query_end_pos=query_end_pos,
                            )

                            _, hybrid_ratio, _, _ = run_with_q_patch_and_capture_image_attention_mode(
                                model,
                                inputs_base_red,
                                q_src=q_mod,
                                layers_to_patch=layer_idxs,
                                mode=mode,
                                pre_img_tokens=args.pre_img_tokens,
                                image_token_count=args.image_token_count,
                                query_end_pos=query_end_pos,
                            )

                            ld_mod_to_hybrid_ratio = run_with_image_attention_ratio_patch_mode(
                                model,
                                inputs_mod_red,
                                tid_a,
                                tid_b,
                                attn_ratio_src=hybrid_ratio,
                                layers_to_patch=layer_idxs,
                                mode=mode,
                                pre_img_tokens=args.pre_img_tokens,
                                image_token_count=args.image_token_count,
                                query_end_pos=query_end_pos,
                            )

                            y = ld_mod_to_hybrid_ratio - ld_mod_to_base_ratio

                            rec = {
                                "condition": "red_circle",
                                "family": "bottom_up",
                                "va": va,
                                "red_idx": int(idx),
                                "mode": mode,
                                "layers": lab,
                                "n_positions": len(positions),
                                "n_image_positions": len(image_positions),
                                "x": float(x),
                                "y": float(y),
                                "ld_base": float(ld_base_red),
                                "ld_mod": float(ld_mod),
                                "ld_mod_to_base_ratio": float(ld_mod_to_base_ratio),
                                "ld_mod_to_hybrid_ratio": float(ld_mod_to_hybrid_ratio),
                                "removed_full_image_attention": float(ld_mod - ld_mod_to_base_ratio),
                                "indirect_image_attention_effect": float(y),
                            }
                            records_by_mode_and_range[mode][lab].append(rec)

                            diff_map = mean_ratio_diff_map_over_setup(
                                hybrid_ratio,
                                base_ratio,
                                layer_idxs,
                            )
                            if diff_map is not None:
                                mean_maps_by_mode_and_range[mode][lab].append(diff_map)

        # save + plot bottom-up prompt
        for mode in patch_modes:
            for lab in layer_labels:
                recs = records_by_mode_and_range[mode][lab]

                out_json = os.path.join(prompt_table_dir, f"{mode}__layers_{lab}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(recs, f, ensure_ascii=False, indent=2)

                n_rot = sum(r["condition"] == "rotation" for r in recs)
                n_red = sum(r["condition"] == "red_circle" for r in recs)
                print(f"[INFO] {prompt_slug} | mode={mode} | layers={lab} | rotation={n_rot} | red_circle={n_red}")

                out_png = os.path.join(prompt_scatter_dir, f"{mode}__layers_{lab}.png")
                plot_effect_scatter(
                    recs,
                    title=f"{model_id} | {prompt_slug}\nBottom-up INDIRECT image-attn effect | {mode} | layers {lab}",
                    save_path=out_png,
                    ylabel="Indirect image-attention effect",
                )

                map_png = os.path.join(prompt_maps_dir, f"{mode}__layers_{lab}.png")
                plot_mean_attention_change_map(
                    mean_maps_by_mode_and_range[mode][lab],
                    title=f"{model_id} | {prompt_slug}\nMean indirect image-attn change | {mode} | layers {lab}",
                    save_path=map_png,
                    image_token_count=args.image_token_count,
                )

        if prompt_slug == bottom_up_prompt_slug_for_combined:
            bottom_up_records_for_combined = records_by_mode_and_range

    # -------------------------
    # B) top-down image-attention effect: x -> r and x -> d
    # -------------------------
    td_prompt_x, td_prefix_x = top_down_prompts["x"]
    td_prompt_r, td_prefix_r = top_down_prompts["r"]
    td_prompt_d, td_prefix_d = top_down_prompts["d"]

    td_slug = (
        slugify(td_prompt_x + td_prefix_x)
        + "__to__"
        + slugify(td_prompt_r + td_prefix_r)
        + "__and__"
        + slugify(td_prompt_d + td_prefix_d)
    )
    td_scatter_dir = os.path.join(scatter_root, td_slug)
    td_table_dir = os.path.join(table_root, td_slug)
    td_maps_dir = os.path.join(maps_root, "top_down", td_slug)
    os.makedirs(td_scatter_dir, exist_ok=True)
    os.makedirs(td_table_dir, exist_ok=True)
    os.makedirs(td_maps_dir, exist_ok=True)

    td_text_x = build_text_prompt(processor, td_prompt_x, td_prefix_x, prompt_format)
    td_text_r = build_text_prompt(processor, td_prompt_r, td_prefix_r, prompt_format)
    td_text_d = build_text_prompt(processor, td_prompt_d, td_prefix_d, prompt_format)

    td_text_x_no_prefix = build_text_prompt(processor, td_prompt_x, None, prompt_format)
    td_text_r_no_prefix = build_text_prompt(processor, td_prompt_r, None, prompt_format)
    td_text_d_no_prefix = build_text_prompt(processor, td_prompt_d, None, prompt_format)

    valid_va_td = []
    for va in va_names:
        b = get_boundary(boundaries, model_id, va, top_down_boundary_slug)
        if b is not None:
            valid_va_td.append(va)

    print(f"\n[INFO] {model_id} | top_down | valid_va = {valid_va_td}")

    td_records_by_mode_and_range: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        mode: {lab: [] for lab in layer_labels} for mode in patch_modes
    }
    td_mean_maps_by_mode_and_range: Dict[str, Dict[str, List[np.ndarray]]] = {
        mode: {lab: [] for lab in layer_labels} for mode in patch_modes
    }

    for va in tqdm(valid_va_td, desc=f"top_down__{td_slug}"):
        center_angle = get_boundary(boundaries, model_id, va, top_down_boundary_slug)
        if center_angle is None:
            continue

        orig_path = os.path.join(duck_rabbit_dir, f"{va}.png")
        if not os.path.exists(orig_path):
            print(f"[WARN] Missing original image: {orig_path}")
            continue

        img_center = make_centered_analysis_image(orig_path, float(center_angle))

        inputs_x = prepare_inputs(processor, img_center, td_text_x, device)
        inputs_r = prepare_inputs(processor, img_center, td_text_r, device)
        inputs_d = prepare_inputs(processor, img_center, td_text_d, device)

        inputs_x_no_prefix = prepare_inputs(processor, img_center, td_text_x_no_prefix, device)
        inputs_r_no_prefix = prepare_inputs(processor, img_center, td_text_r_no_prefix, device)
        inputs_d_no_prefix = prepare_inputs(processor, img_center, td_text_d_no_prefix, device)

        query_end_pos_x = int(inputs_x_no_prefix["input_ids"].shape[1])
        query_end_pos_r = int(inputs_r_no_prefix["input_ids"].shape[1])
        query_end_pos_d = int(inputs_d_no_prefix["input_ids"].shape[1])

        # precompute x-base ratios for each mode
        x_base_ratio_by_mode = {}
        for mode in patch_modes:
            _, _, ratio_x, _, _ = run_and_capture_image_attention_mode(
                model,
                inputs_x,
                tid_a,
                tid_b,
                mode=mode,
                pre_img_tokens=args.pre_img_tokens,
                image_token_count=args.image_token_count,
                query_end_pos=query_end_pos_x,
            )
            x_base_ratio_by_mode[mode] = ratio_x

        td_pairs = [
            ("x_to_r", inputs_x, inputs_r, query_end_pos_x, query_end_pos_r),
            ("x_to_d", inputs_x, inputs_d, query_end_pos_x, query_end_pos_d),
        ]

        for direction, inputs_base, inputs_mod, qend_base, qend_mod in td_pairs:
            ld_base = next_token_logitdiff_from_inputs(model, inputs_base, tid_a, tid_b)

            for mode in patch_modes:
                ld_mod, _, ratio_mod, positions, image_positions = run_and_capture_image_attention_mode(
                    model,
                    inputs_mod,
                    tid_a,
                    tid_b,
                    mode=mode,
                    pre_img_tokens=args.pre_img_tokens,
                    image_token_count=args.image_token_count,
                    query_end_pos=qend_mod,
                )
                x = ld_mod - ld_base

                base_ratio = x_base_ratio_by_mode[mode]

                for lab, layer_idxs in zip(layer_labels, layer_lists):
                    ld_mod_to_base_ratio = run_with_image_attention_ratio_patch_mode(
                        model,
                        inputs_mod,
                        tid_a,
                        tid_b,
                        attn_ratio_src=base_ratio,
                        layers_to_patch=layer_idxs,
                        mode=mode,
                        pre_img_tokens=args.pre_img_tokens,
                        image_token_count=args.image_token_count,
                        query_end_pos=qend_mod,
                    )
                    y = ld_mod - ld_mod_to_base_ratio

                    rec = {
                        "condition": "top_down",
                        "family": "top_down",
                        "direction": direction,
                        "va": va,
                        "mode": mode,
                        "layers": lab,
                        "n_positions": len(positions),
                        "n_image_positions": len(image_positions),
                        "x": float(x),
                        "y": float(y),
                        "ld_base": float(ld_base),
                        "ld_mod": float(ld_mod),
                        "ld_mod_to_base_ratio": float(ld_mod_to_base_ratio),
                        "image_attention_effect": float(y),
                    }
                    td_records_by_mode_and_range[mode][lab].append(rec)

                    diff_map = mean_ratio_diff_map_over_setup(
                        ratio_mod,
                        base_ratio,
                        layer_idxs,
                    )
                    if diff_map is not None:
                        td_mean_maps_by_mode_and_range[mode][lab].append(diff_map)

    # save + plot top-down
    for mode in patch_modes:
        for lab in layer_labels:
            recs = td_records_by_mode_and_range[mode][lab]

            out_json = os.path.join(td_table_dir, f"{mode}__layers_{lab}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(recs, f, ensure_ascii=False, indent=2)

            n_td = sum(r["condition"] == "top_down" for r in recs)
            print(f"[INFO] top_down | mode={mode} | layers={lab} | n={n_td}")

            out_png = os.path.join(td_scatter_dir, f"{mode}__layers_{lab}.png")
            plot_effect_scatter(
                recs,
                title=f"{model_id} | Top-down x→r/d | {mode} | layers {lab}",
                save_path=out_png,
                ylabel="Effect explained by image attention",
            )

            map_png = os.path.join(td_maps_dir, f"{mode}__layers_{lab}.png")
            plot_mean_attention_change_map(
                td_mean_maps_by_mode_and_range[mode][lab],
                title=f"{model_id} | Top-down mean image-attn change | {mode} | layers {lab}",
                save_path=map_png,
                image_token_count=args.image_token_count,
            )

    # -------------------------
    # C) combined bottom-up vs top-down + through-origin fits
    # -------------------------
    combined_scatter_dir = os.path.join(scatter_root, "combined_bottomup_vs_topdown")
    combined_table_dir = os.path.join(table_root, "combined_bottomup_vs_topdown")
    os.makedirs(combined_scatter_dir, exist_ok=True)
    os.makedirs(combined_table_dir, exist_ok=True)

    if bottom_up_records_for_combined is None:
        print("[WARN] No bottom-up 'I see a' records found for combined plot.")
    else:
        for mode in patch_modes:
            for lab in layer_labels:
                recs_bottom = bottom_up_records_for_combined[mode][lab]
                recs_top = td_records_by_mode_and_range[mode][lab]
                recs_combined = recs_bottom + recs_top

                out_json = os.path.join(combined_table_dir, f"{mode}__layers_{lab}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(recs_combined, f, ensure_ascii=False, indent=2)

                fit_stats = plot_combined_effect_scatter_with_fits(
                    recs_bottom,
                    recs_top,
                    #     f"{model_id} | bottom-up (indirect) vs top-down | I see a\n"
                    #     f"{mode} | layers {lab}"
                    save_path=os.path.join(combined_scatter_dir, f"{mode}__layers_{lab}.png"),
                )

                out_fit_json = os.path.join(combined_table_dir, f"{mode}__layers_{lab}__fit.json")
                with open(out_fit_json, "w", encoding="utf-8") as f:
                    json.dump(fit_stats, f, ensure_ascii=False, indent=2)

                n_rot = sum(r["condition"] == "rotation" for r in recs_combined)
                n_red = sum(r["condition"] == "red_circle" for r in recs_combined)
                n_td = sum(r["condition"] == "top_down" for r in recs_combined)
                print(
                    f"[INFO] combined | mode={mode} | layers={lab} | "
                    f"rotation={n_rot} | red_circle={n_red} | top_down={n_td} | "
                    f"beta_bottom={fit_stats['bottom_up']['slope']:.3f} | "
                    f"beta_top={fit_stats['top_down']['slope']:.3f}"
                )

    del model, processor
    torch.cuda.empty_cache()
