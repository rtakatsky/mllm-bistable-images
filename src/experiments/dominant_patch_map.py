import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json
import argparse
import random
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def dominant_patch_labels_and_counts(dominant_patch_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    dominant_patch_map: [P_img, T], zeros below threshold.
    Returns:
      best_idx: [P_img] argmax token
      valid:    [P_img] best_val > 0
      counts:   [T] number of dominant patches per token
    """
    best_idx = np.argmax(dominant_patch_map, axis=1)
    best_val = np.max(dominant_patch_map, axis=1)
    valid = best_val > 0
    T = dominant_patch_map.shape[1]
    counts = np.array([(valid & (best_idx == t)).sum() for t in range(T)], dtype=int)
    return best_idx, valid, counts

def scale_xy_to_model(center_xy_raw, img: Image.Image, model_size: int = 336):
    """
    Convert (x,y) in raw image pixels -> coordinates in model_size x model_size space.
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return center_xy_raw
    sx = model_size / float(w)
    sy = model_size / float(h)
    return (center_xy_raw[0] * sx, center_xy_raw[1] * sy)

def patch_centers_px(image_size: int = 336, patch_size: int = 14, n_patches: Optional[int] = None) -> np.ndarray:
    grid = image_size // patch_size
    if n_patches is None:
        n_patches = grid * grid
    coords = []
    for p in range(n_patches):
        r = p // grid
        c = p % grid
        x = c * patch_size + patch_size / 2.0
        y = r * patch_size + patch_size / 2.0
        coords.append([x, y])
    return np.asarray(coords, dtype=np.float32)


def dominant_patch_coords(dominant_patch_map: np.ndarray, image_size: int = 336, patch_size: int = 14) -> np.ndarray:
    _, valid, _ = dominant_patch_labels_and_counts(dominant_patch_map)
    coords_all = patch_centers_px(image_size=image_size, patch_size=patch_size, n_patches=dominant_patch_map.shape[0])
    return coords_all[valid]


def detect_red_circle_center(
    image: Image.Image,
    red_thr: int = 180,
    green_thr: int = 120,
    blue_thr: int = 120,
) -> Optional[Tuple[float, float]]:
    arr = np.asarray(image.convert("RGB"))
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)

    mask = (r >= red_thr) & (g <= green_thr) & (b <= blue_thr) & ((r - g) > 40) & ((r - b) > 40)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def distances_from_center_to_coords(center_xy: Tuple[float, float], coords_xy: np.ndarray) -> np.ndarray:
    if coords_xy is None or len(coords_xy) == 0:
        return np.array([], dtype=np.float32)
    c = np.asarray(center_xy, dtype=np.float32)[None, :]
    d = np.sqrt(np.sum((coords_xy - c) ** 2, axis=1))
    return d.astype(np.float32)


def first_token_score_diff(result_item: Dict[str, Any], token_a: str, token_b: str) -> float:
    outs = result_item.get("output", [])
    if len(outs) == 0:
        return np.nan
    d = outs[0]
    toks = d.get("tokens", [])
    logits = d.get("logits", None)
    probs = d.get("probs", None)

    lmap = dict(zip(toks, logits)) if logits is not None else {}
    pmap = dict(zip(toks, probs)) if probs is not None else {}

    if token_a in lmap and token_b in lmap:
        return float(lmap[token_a] - lmap[token_b])
    if token_a in pmap and token_b in pmap:
        return float(pmap[token_a] - pmap[token_b])
    return np.nan


@torch.no_grad()
def compute_dominant_patch_map_for_image(
    *,
    model,
    processor,
    image: Image.Image,
    text_input: str,
    tokens_of_interest_du_rabb: Dict[str, int],  # 2 tokens
    get_cache_fn,
    logitlens_fn,
    dominant_patch_map_from_logitlens_fn,
    device,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    NO cache saving: save_dir=None
    Returns dominant_patch_map [P_img, 2]
    """
    cache = get_cache_fn(model, processor, image, text_input, device=device, save_dir=None, cache_keys=["resid_post"])

    hs = cache["resid_post"]
    seq_len = hs[0].shape[-2] if isinstance(hs, (list, tuple)) else hs.shape[-2]

    logits_softmax = logitlens_fn(
        model,
        hs,
        tokens_of_interest_du_rabb,
        pos=list(range(seq_len)),
        device=device,
        softmax=True,
        save_path=None,
        ln_mode="independent",
    )

    # Base-view image positions from the expanded input_ids (correct for
    # every prompt template; AnyRes tiles excluded) instead of range(5, 581).
    from utils import base_view_image_positions
    img_pos = base_view_image_positions(processor, model, text_input, image)
    dpm = dominant_patch_map_from_logitlens_fn(
        logits_softmax,
        tokens_of_interest=tokens_of_interest_du_rabb,
        threshold=threshold,
        img_pos=img_pos,
    )
    return dpm


# =========================
# Plotting
# =========================

#     plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
#     plt.close(fig)

def plot_scatter_counts_vs_diff(records, token_a, token_b, title: Optional[str] = None, save_path: str = None):
    if len(records) == 0:
        print("[WARN] No records to plot scatter.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    diffs = np.array([r["score_diff"] for r in records if np.isfinite(r["score_diff"])], dtype=float)
    vmax = float(np.nanmax(np.abs(diffs))) if len(diffs) else 1.0
    vmax = max(vmax, 1e-6)

    last_sc = None

    # rotation: x marker (filled by colormap)
    sub = [r for r in records if r["condition"] == "rotation" and np.isfinite(r["score_diff"])]
    if sub:
        xs = np.array([r["n_dom_a"] for r in sub], dtype=float)
        ys = np.array([r["n_dom_b"] for r in sub], dtype=float)
        cs = np.array([r["score_diff"] for r in sub], dtype=float)
        last_sc = ax.scatter(
            xs, ys, c=cs, cmap="coolwarm", vmin=-vmax, vmax=vmax,
            s=40, alpha=0.9, marker="x",
            linewidths=1.2,
            label="rotation",
        )

    # red_circle: empty circles, edge colored by colormap
    sub = [r for r in records if r["condition"] == "red_circle" and np.isfinite(r["score_diff"])]
    if sub:
        xs = np.array([r["n_dom_a"] for r in sub], dtype=float)
        ys = np.array([r["n_dom_b"] for r in sub], dtype=float)
        cs = np.array([r["score_diff"] for r in sub], dtype=float)
        last_sc = ax.scatter(
            xs, ys, c=cs, cmap="coolwarm", vmin=-vmax, vmax=vmax,
            s=40, alpha=0.9, marker="o",
            facecolors="none", linewidths=1.0,
            label="red_circle",
        )

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=ax)
        cbar.set_label(f'Δ({token_a} - {token_b})')

    ax.set_xlabel(f"# dominant patches ({token_a})")
    ax.set_ylabel(f"# dominant patches ({token_b})")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_distance_histogram(d_red: np.ndarray, d_base: np.ndarray, radius_px: Optional[float], title: str, save_path: str, bins: int = 40):
    fig, ax = plt.subplots(figsize=(8, 5))

    d_red = np.asarray(d_red, dtype=float)
    d_base = np.asarray(d_base, dtype=float)
    d_red = d_red[np.isfinite(d_red)]
    d_base = d_base[np.isfinite(d_base)]

    all_vals = np.concatenate([d_red, d_base]) if (len(d_red) + len(d_base)) else np.array([0, 1])
    vmax = float(np.max(all_vals)) if len(all_vals) else 1.0
    vmax = max(vmax, 1.0)
    bins_arr = np.linspace(0, vmax, bins + 1)

    ax.hist(d_base, bins=bins_arr, alpha=0.45, label="No red circle (baseline)", density=True)
    ax.hist(d_red,  bins=bins_arr, alpha=0.45, label="With red circle", density=True)

    if radius_px is not None and np.isfinite(radius_px):
        ax.axvline(radius_px, linestyle="--", linewidth=1.2, color="k", label=f"Circle radius ({radius_px:.1f}px)")

    ax.set_xlabel("Distance from red-circle center to dominant patch (px)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# =========================
# Main
# =========================

if __name__ == "__main__":
    from utils import *  # expects: load_vlm, slugify, get_cache, logitlens, dominant_patch_map_from_logitlens, RotatedRGBAImageDataset, RedCircleImageDataset, run_vlm_sampling

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--base_dir", type=str, default=PROJECT_ROOT)
    parser.add_argument("--rotation_offset", type=int, default=5)      # [-5..5]
    parser.add_argument("--dominant_threshold", type=float, default=0.1)
    parser.add_argument("--radius_ratio", type=float, default=1/14)
    parser.add_argument("--stride_ratio", type=float, default=1/7)
    parser.add_argument("--line_width", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)                # for red-circle sampling
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading/inference; replot from outputs/.../cache/<model>/ (CPU-friendly).")
    args = parser.parse_args()

    model_id = args.model
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = args.base_dir
    data_dir_orig = os.path.join(base_dir, "data/duck_rabbit")
    boundaries_path = os.path.join(data_dir_orig, "boundaries.json")            # <-- your filename

    results_dir = os.path.join(base_dir, "outputs/dominant_patch_map")
    os.makedirs(results_dir, exist_ok=True)

    prompts = [
        # ["List every animal in the image, each in one word.", "There's a"],
        ["List every animal in the image, each in one word.", "I see a"],
    ]

    vlm_dict = {
        "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
        "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
        "llava-v1.6-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava-v1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llama3-llava-next-8b": "llava-hf/llama3-llava-next-8b-hf",
    }

    with open(boundaries_path, "r", encoding="utf-8") as f:
        boundaries = json.load(f)

    # Pick va_* that have *some* boundary (we still check per prompt)
    # Expanded pipeline-generated set, deterministic order (va_set_manifest.json);
    # partial runs therefore cover a well-defined "first N" subset.
    with open(os.path.join(DATA_DIR, "duck_rabbit", "va_set_manifest.json")) as f:
        all_va_names = [f"va_s{seed:03d}" for seed in json.load(f)["seeds"]]
    valid_va_names = []
    for va in all_va_names:
        # accept if any schema yields a non-null boundary for at least one prompt (best-effort)
        has_any = False
        for pr, px in prompts:
            ps = slugify(pr + px)
            b = get_boundary(boundaries, model_name=model_id, image_name=va, prompt_slug=ps)
            if b is not None:
                has_any = True
                break
        if has_any:
            valid_va_names.append(va)
    print(f"[INFO] va with some boundary for {model_id}: {valid_va_names}")

    tok_a_str, tok_b_str = get_duck_rabbit_tokens(model_id)
    if args.plot_only:
        processor = model = prompt_format = None
        tokens_of_interest_du_rabb = None
        print("[PLOT-ONLY] skipping model load; replotting from cache")
    else:
        print(f"Loading {model_id}...")
        processor, model, prompt_format = load_vlm(vlm_dict[model_id], return_prompt_format=True)
        model.to(device).eval()
        tokenizer = processor.tokenizer
        tokens_of_interest_du_rabb = {
            tok_a_str: tokenizer.convert_tokens_to_ids(tok_a_str),
            tok_b_str: tokenizer.convert_tokens_to_ids(tok_b_str),
        }

    # bottom-up score extraction setup (first-token logits)
    sample_n = 1
    temperature = 1.0
    max_new_tokens = 1
    seed = 42
    batch_size = 4
    top_k = 100
    use_logit = True

    # storage
    all_records = []
    all_d_red = []
    all_d_base = []

    scatter_dir = os.path.join(results_dir, "plots", model_id, "scatter")
    hist_dir    = os.path.join(results_dir, "plots", model_id, "hist")

    for prompt, prompt_prefix in prompts:
        prompt_slug = slugify(prompt + prompt_prefix)
        text_input = f"USER: <image>\n{prompt} ASSISTANT:{prompt_prefix}"

        prompt_records = []
        prompt_d_red = []
        prompt_d_base = []

        for va_name in tqdm([] if args.plot_only else valid_va_names, desc=f"{prompt_slug}"):
            center_angle = get_boundary(boundaries, model_name=model_id, image_name=va_name, prompt_slug=prompt_slug)
            if center_angle is None:
                continue

            # rotation angles
            rot_angles = [int(center_angle) + da for da in range(-args.rotation_offset, args.rotation_offset + 1)]
            K = len(rot_angles)

            # -------------------------
            # A) rotation condition
            # -------------------------
            img_path_orig = os.path.join(data_dir_orig, f"{va_name}.png")
            if not os.path.exists(img_path_orig):
                continue

            rot_dataset = RotatedRGBAImageDataset(img_path_orig, rot_angles)
            rot_results = run_vlm_sampling(
                model, processor, rot_dataset, prompt,
                sample_n=sample_n, temperature=temperature, max_new_tokens=max_new_tokens,
                seed=seed, batch_size=batch_size,
                use_logit=use_logit, top_k=top_k,
                prompt_format=prompt_format,
                prompt_prefix=prompt_prefix,
                extra_tokens=[tok_a_str, tok_b_str],
            )

            for idx, item in enumerate(rot_results):
                angle = int(item["angle"])
                img = rot_dataset[idx]

                dpm = compute_dominant_patch_map_for_image(
                    model=model,
                    processor=processor,
                    image=img,
                    text_input=text_input,
                    tokens_of_interest_du_rabb=tokens_of_interest_du_rabb,
                    get_cache_fn=get_cache,
                    logitlens_fn=logitlens,
                    dominant_patch_map_from_logitlens_fn=dominant_patch_map_from_logitlens,
                    device=device,
                    threshold=args.dominant_threshold,
                )
                _, _, counts = dominant_patch_labels_and_counts(dpm)
                score_diff = first_token_score_diff(item, tok_a_str, tok_b_str)

                rec = {
                    "condition": "rotation",
                    "image_name": va_name,
                    "prompt_slug": prompt_slug,
                    "angle": angle,
                    "n_dom_a": int(counts[0]),
                    "n_dom_b": int(counts[1]),
                    "score_diff": float(score_diff),
                }
                prompt_records.append(rec)
                all_records.append(rec)

            # -------------------------
            # B) red-circle condition (sample ~K random placements)
            # -------------------------
            # Construct the centered VA image in-memory from the raw source +
            # saved boundary, rather than reading a pre-centered .png.
            centered_va = centered_va_image(
                raw_image_path=img_path_orig,
                boundaries=boundaries,
                model_name=model_id,
                image_name=va_name,
                prompt_slug=prompt_slug,
            )
            if centered_va is None:
                continue
            base_img = centered_va.convert("RGB")

            dpm_base = compute_dominant_patch_map_for_image(
                model=model,
                processor=processor,
                image=base_img,
                text_input=text_input,
                tokens_of_interest_du_rabb=tokens_of_interest_du_rabb,
                get_cache_fn=get_cache,
                logitlens_fn=logitlens,
                dominant_patch_map_from_logitlens_fn=dominant_patch_map_from_logitlens,
                device=device,
                threshold=args.dominant_threshold,
            )
            base_dom_coords = dominant_patch_coords(dpm_base)

            red_dataset = RedCircleImageDataset(
                centered_va,
                radius_ratio=args.radius_ratio,
                stride_ratio=args.stride_ratio,
                line_width=args.line_width,
            )
            if len(red_dataset) == 0:
                continue

            k_red = min(K, len(red_dataset))
            red_indices = random.sample(range(len(red_dataset)), k_red)

            # tiny wrapper so run_vlm_sampling can read dataset.angles
            class _SubsetDataset(torch.utils.data.Dataset):
                def __init__(self, parent, indices):
                    self.parent = parent
                    self.indices = indices
                    self.angles = indices  # placeholder used as "angle"
                def __len__(self):
                    return len(self.indices)
                def __getitem__(self, i):
                    return self.parent[self.indices[i]]

            red_subset = _SubsetDataset(red_dataset, red_indices)

            red_results = run_vlm_sampling(
                model, processor, red_subset, prompt,
                sample_n=sample_n, temperature=temperature, max_new_tokens=max_new_tokens,
                seed=seed, batch_size=batch_size,
                use_logit=use_logit, top_k=top_k,
                prompt_format=prompt_format,
                prompt_prefix=prompt_prefix,
                extra_tokens=[tok_a_str, tok_b_str],
            )

            for local_i, item in enumerate(red_results):
                ds_idx = red_indices[local_i]
                img_red = red_dataset[ds_idx]

                center_xy_raw = detect_red_circle_center(img_red)
                if center_xy_raw is None:
                    continue

                MODEL_SIZE = 336  # must match your patch coord convention
                center_xy = scale_xy_to_model(center_xy_raw, img_red, model_size=MODEL_SIZE)

                dpm_red = compute_dominant_patch_map_for_image(
                    model=model,
                    processor=processor,
                    image=img_red,
                    text_input=text_input,
                    tokens_of_interest_du_rabb=tokens_of_interest_du_rabb,
                    get_cache_fn=get_cache,
                    logitlens_fn=logitlens,
                    dominant_patch_map_from_logitlens_fn=dominant_patch_map_from_logitlens,
                    device=device,
                    threshold=args.dominant_threshold,
                )
                _, _, counts = dominant_patch_labels_and_counts(dpm_red)
                score_diff = first_token_score_diff(item, tok_a_str, tok_b_str)

                rec = {
                    "condition": "red_circle",
                    "image_name": va_name,
                    "prompt_slug": prompt_slug,
                    "red_idx": int(ds_idx),
                    "n_dom_a": int(counts[0]),
                    "n_dom_b": int(counts[1]),
                    "score_diff": float(score_diff),
                }
                prompt_records.append(rec)
                all_records.append(rec)

                # histogram distances: use same detected centers for both conditions
                red_dom_coords = dominant_patch_coords(dpm_red)
                d_red  = distances_from_center_to_coords(center_xy, red_dom_coords)
                d_base = distances_from_center_to_coords(center_xy, base_dom_coords)

                if len(d_red):
                    prompt_d_red.append(d_red)
                    all_d_red.append(d_red)
                if len(d_base):
                    prompt_d_base.append(d_base)
                    all_d_base.append(d_base)

        # ---- plot-input cache (every paper figure must be replottable without GPU) ----
        cache_dir = os.path.join(results_dir, "cache", model_id)
        os.makedirs(cache_dir, exist_ok=True)
        rec_path = os.path.join(cache_dir, f"{prompt_slug}__records.json")
        dist_path = os.path.join(cache_dir, f"{prompt_slug}__distances.npz")
        if args.plot_only:
            if not os.path.exists(rec_path):
                print(f"[PLOT-ONLY SKIP] missing cache: {rec_path}")
                continue
            with open(rec_path, "r", encoding="utf-8") as f:
                prompt_records = json.load(f)
            all_records.extend(prompt_records)
            dz = np.load(dist_path)
            prompt_d_red = [dz["d_red"]] if len(dz["d_red"]) else []
            prompt_d_base = [dz["d_base"]] if len(dz["d_base"]) else []
            all_d_red.extend(prompt_d_red)
            all_d_base.extend(prompt_d_base)
        else:
            with open(rec_path, "w", encoding="utf-8") as f:
                json.dump(prompt_records, f, indent=1)
            np.savez(dist_path,
                     d_red=np.concatenate(prompt_d_red) if len(prompt_d_red) else np.array([]),
                     d_base=np.concatenate(prompt_d_base) if len(prompt_d_base) else np.array([]))

        # prompt-level plots
        out_scatter = os.path.join(scatter_dir, f"{prompt_slug}.png")
        plot_scatter_counts_vs_diff(
            records=prompt_records,
            token_a=tok_a_str,
            token_b=tok_b_str,
            title=None,
            save_path=out_scatter,
        )

        d_red_cat = np.concatenate(prompt_d_red) if len(prompt_d_red) else np.array([])
        d_base_cat = np.concatenate(prompt_d_base) if len(prompt_d_base) else np.array([])
        radius_px = 336 * float(args.radius_ratio)

        out_hist = os.path.join(hist_dir, f"{prompt_slug}_distance_hist.png")
        plot_distance_histogram(
            d_red=d_red_cat,
            d_base=d_base_cat,
            radius_px=radius_px,
            title=None,
            save_path=out_hist,
        )

    # all-prompts plots
    out_scatter_all = os.path.join(results_dir, "plots", model_id, "all_prompts_scatter.png")
    plot_scatter_counts_vs_diff(
        records=all_records,
        token_a=tok_a_str,
        token_b=tok_b_str,
        save_path=out_scatter_all,
    )

    d_red_all = np.concatenate(all_d_red) if len(all_d_red) else np.array([])
    d_base_all = np.concatenate(all_d_base) if len(all_d_base) else np.array([])
    radius_px = 336 * float(args.radius_ratio)

    out_hist_all = os.path.join(results_dir, "plots", model_id, "all_prompts_distance_hist.png")
    plot_distance_histogram(
        d_red=d_red_all,
        d_base=d_base_all,
        radius_px=radius_px,
        title=None,
        save_path=out_hist_all,
    )

    del model, processor
    torch.cuda.empty_cache()