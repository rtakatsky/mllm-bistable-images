import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json
import argparse
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from tqdm import tqdm

from utils import *


def build_prompt(prompt_text: str, prompt_prefix: str, prompt_format: Optional[str]) -> str:
    if prompt_format is None:
        return f"USER: <image>\n{prompt_text} ASSISTANT:{prompt_prefix}"
    text_prompt = prompt_format.format(prompt=prompt_text)
    if prompt_prefix is not None:
        text_prompt = text_prompt + prompt_prefix
    return text_prompt


def token_id_or_warn(tokenizer, token_str: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token_str)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and tid == unk_id:
        print(f"[WARN] Token '{token_str}' mapped to unk_token_id={unk_id}.")
    return int(tid)


def adapt_token(tokenizer, token_str: str) -> str:
    """Swap SentencePiece '▁' vs BPE 'Ġ' word-boundary prefix to match the
    tokenizer's vocab (llama3-llava-next-8b uses GPT-2-style BPE)."""
    tid = tokenizer.convert_tokens_to_ids(token_str)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk_id is not None and tid == unk_id):
        alt = token_str.replace("▁", "Ġ") if "▁" in token_str else token_str.replace("Ġ", "▁")
        alt_tid = tokenizer.convert_tokens_to_ids(alt)
        if alt_tid is not None and alt_tid != unk_id:
            return alt
    return token_str


def get_image_token_positions(
    seq_len: int,
    pre_img_tokens: int = 5,
    image_token_count: int = (336 // 14) ** 2,
) -> List[int]:
    end = min(seq_len, pre_img_tokens + image_token_count)
    return list(range(pre_img_tokens, end))


def logit_diff_from_vec(
    vec: np.ndarray,
    token_to_idx: Dict[str, int],
    pos_token: str,
    neg_token: str,
) -> float:
    return float(vec[token_to_idx[pos_token]] - vec[token_to_idx[neg_token]])

def compute_source_patch_styles(
    *,
    model,
    processor,
    source_image: Image.Image,
    source_prompt: str,
    device: str,
    bird_tok: str,
    rabb_tok: str,
    pre_img_tokens: int = 5,
    image_token_count: int = (336 // 14) ** 2,
    cache_save_dir: Optional[str] = None,
    threshold: float = 0.1,   # NEW
):
    """
    For each source image patch token:
      - color label: bird / rabbit / none
      - marker label: side / two / none

    Dominance rule:
      - if both compared tokens are below threshold -> none
      - else pick the larger one
    """
    tokenizer = processor.tokenizer
    side_tok = adapt_token(tokenizer, "▁side")
    two_tok = adapt_token(tokenizer, "▁two")
    style_tokens = {
        bird_tok: token_id_or_warn(tokenizer, bird_tok),
        rabb_tok: token_id_or_warn(tokenizer, rabb_tok),
        side_tok: token_id_or_warn(tokenizer, side_tok),
        two_tok: token_id_or_warn(tokenizer, two_tok),
    }

    cache = get_cache(
        model=model,
        processor=processor,
        image=source_image,
        prompt=source_prompt,
        device=device,
        save_dir=cache_save_dir,
        cache_keys=["resid_post"],
    )

    seq_len = int(cache["resid_post"].shape[1])
    pos = list(range(seq_len))

    probs = logitlens(
        model=model,
        hidden_states=cache["resid_post"],
        tokens_of_interest=style_tokens,
        pos=pos,
        softmax=True,
        head_out=False,
        device=device,
        ln_mode="independent",
    )  # [L+1, S, T]

    # Base-view image positions from the expanded input_ids (correct for
    # every prompt template; AnyRes tiles excluded) instead of range(5, 581).
    from utils import base_view_image_positions
    img_positions = base_view_image_positions(processor, model, source_prompt, source_image,
                                              n_base=image_token_count)
    img_probs = probs[:, img_positions, :]     # [L+1, Nimg, 4]
    max_probs = img_probs.max(axis=0)          # [Nimg, 4]

    token_names = list(style_tokens.keys())
    idx_bird = token_names.index(bird_tok)
    idx_rabb = token_names.index(rabb_tok)
    idx_side = token_names.index(side_tok)
    idx_two = token_names.index(two_tok)

    color_labels = []
    marker_labels = []

    for p in range(max_probs.shape[0]):
        bird_score = float(max_probs[p, idx_bird])
        rabb_score = float(max_probs[p, idx_rabb])
        side_score = float(max_probs[p, idx_side])
        two_score = float(max_probs[p, idx_two])

        # color channel: bird vs rabbit with threshold
        if max(bird_score, rabb_score) < threshold:
            color_labels.append("none")
        else:
            color_labels.append("bird" if bird_score >= rabb_score else "rabbit")

        # marker channel: side vs two with threshold
        if max(side_score, two_score) < threshold:
            marker_labels.append("none")
        else:
            marker_labels.append("two")

    return {
        "img_positions": img_positions,
        "color_labels": color_labels,
        "marker_labels": marker_labels,
    }

def plot_cross_task_scatter(
    records: List[Dict[str, Any]],
    save_path: str,
    title: Optional[str] = None,
    jitter_frac: float = 0.003,
    jitter_seed: int = 0,
):
    if len(records) == 0:
        print("[WARN] No records to plot.")
        return

    color_map = {
        "bird": "tab:blue",
        "rabbit": "tab:orange",
        "none": "black",
    }
    marker_map = {
        # "side": "x",
        "two": "*",
        "none": "o",
    }

    xs_all = np.array([r["x"] for r in records if np.isfinite(r["x"])], dtype=float)
    ys_all = np.array([r["y"] for r in records if np.isfinite(r["y"])], dtype=float)
    x_span = float(np.ptp(xs_all)) if len(xs_all) else 1.0
    y_span = float(np.ptp(ys_all)) if len(ys_all) else 1.0
    x_jitter = max(x_span, 1e-6) * jitter_frac
    y_jitter = max(y_span, 1e-6) * jitter_frac

    rng = np.random.default_rng(jitter_seed)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot order matters: draw "two" last so it stands out.
    plot_order = [
        # ("bird", "side"),
        # ("rabbit", "side"),
        # ("none", "side"),
        ("bird", "none"),
        ("rabbit", "none"),
        ("none", "none"),
        ("bird", "two"),
        ("rabbit", "two"),
        ("none", "two"),
    ]

    for color_label, marker_label in plot_order:
        sub = [
            r for r in records
            if r["color_label"] == color_label and r["marker_label"] == marker_label
            and np.isfinite(r["x"]) and np.isfinite(r["y"])
        ]
        if not sub:
            continue

        xs = np.array([r["x"] for r in sub], dtype=float)
        ys = np.array([r["y"] for r in sub], dtype=float)

        if jitter_frac > 0:
            xs_plot = xs + rng.normal(0.0, x_jitter, size=len(xs))
            ys_plot = ys + rng.normal(0.0, y_jitter, size=len(ys))
        else:
            xs_plot = xs
            ys_plot = ys

        base_color = color_map[color_label]
        marker = marker_map[marker_label]

        # "two" always red
        draw_color = "red" if marker_label == "two" else base_color

        if marker == "x":
            ax.scatter(
                xs_plot, ys_plot,
                marker=marker,
                c=draw_color,
                s=28,
                alpha=0.65,
                linewidths=1.0,
                zorder=2,
            )
        elif marker == "o":
            # normal filled dot for below-threshold side/two
            ax.scatter(
                xs_plot, ys_plot,
                marker=marker,
                c=draw_color,
                s=20,
                alpha=0.65,
                linewidths=0.0,
                zorder=2,
            )
        else:  # star for "two"
            ax.scatter(
                xs_plot, ys_plot,
                marker=marker,
                c=draw_color,
                s=55,
                alpha=0.9,
                linewidths=0.6,
                zorder=3,
            )

    ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axvline(0.0, linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_xlabel('Count effect: Δ(logit("2") - logit("1"))')
    ax.set_ylabel('Punct effect: Δ(logit(" and") - logit("."))')
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)

    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='bird-dominant',
               markerfacecolor='tab:blue', markeredgecolor='tab:blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='rabbit-dominant',
               markerfacecolor='tab:orange', markeredgecolor='tab:orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='below bird/rabbit threshold',
               markerfacecolor='black', markeredgecolor='black', markersize=8),
        # Line2D([0], [0], marker='x', color='black', label='"side" dominant',
        Line2D([0], [0], marker='*', color='red', label='"two" dominant',
               linestyle='None', markersize=10),
    ]
    ax.legend(handles=legend_items, loc="best")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def model_resampling_ablation(
    model,
    processor,
    source_image,
    target_image,
    source_prompt,
    target_prompt,
    tokens_of_interest,
    device="cpu",
    save_dir=None,
    ablation_keys=["resid_post"],
    positions_to_ablate: Optional[List[int]] = None,
    resid_post_layers_to_run: Optional[List[int]] = None,
    append_token_ids: Optional[List[int]] = None,
):
    """
    Reuses your original structure, but for this experiment:
      - only use resid_post
      - only need source_cache keys: embed, resid_post
    """
    model = model.to(device)
    model.eval()

    llm = model.language_model.model
    results = {}

    source_inputs = processor(text=source_prompt, images=source_image, return_tensors="pt").to(device)
    target_inputs = processor(text=target_prompt, images=target_image, return_tensors="pt").to(device)

    # Some chat templates end the assistant turn with a merged whitespace token
    # (Llama-3: "\n\n" -> ĊĊ), so no *string* prefix can put the readout at the
    # position the model actually answers from: appending "\n" re-merges into
    # ĊĊĊ, and appending " " forces an off-path Ġ. append_token_ids therefore
    # continues the turn at the token level, after image expansion (appending at
    # the end leaves the image-token positions untouched).
    if append_token_ids:
        extra = torch.tensor([append_token_ids], device=device)
        for inputs in (source_inputs, target_inputs):
            inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(extra)], dim=1)

    S_src = source_inputs["input_ids"].shape[1]
    S_tgt = target_inputs["input_ids"].shape[1]
    assert S_src == S_tgt, "Source and target sequence lengths must match for resampling ablation"
    print(f"Source sequence length: {S_src}, Target sequence length: {S_tgt}")

    interesting_ids = torch.tensor(list(tokens_of_interest.values()), device=device)

    def run_forward(inputs):
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]
            logits = model.language_model.lm_head(last_hidden)
            return logits[0, -1, interesting_ids].detach().cpu().numpy()

    results["original"] = run_forward(target_inputs)

    # only request what we need to avoid OOM
    source_cache = get_cache(
        model,
        processor,
        source_image,
        source_prompt,
        device=device,
        save_dir=save_dir,
        cache_keys=["embed", "resid_post"],
    )
    for key in source_cache.keys():
        source_cache[key] = source_cache[key].to(device)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "original.npy"), results["original"])

    L = len(llm.layers)
    S = S_src
    T = len(interesting_ids)

    token_positions = list(range(S)) if positions_to_ablate is None else list(positions_to_ablate)
    resid_post_layers_to_run = list(range(L + 1)) if resid_post_layers_to_run is None else list(resid_post_layers_to_run)

    def resample_token_out(out, t_idx, cached):
        if isinstance(out, tuple):
            y = out[0].clone()
            y[:, t_idx, :] = cached
            return (y,) + out[1:]
        else:
            y = out.clone()
            y[:, t_idx, :] = cached
            return y

    def resample_token_inp(inp, t_idx, cached):
        x = inp[0].clone()
        x[:, t_idx, :] = cached
        return (x,) + inp[1:]

    for ablation_key in ablation_keys:
        print(f"Ablating {ablation_key}")

        if save_dir is not None and os.path.exists(os.path.join(save_dir, f"{ablation_key}.npy")):
            print(f"Loading {ablation_key}.npy")
            ablated_store = np.load(os.path.join(save_dir, f"{ablation_key}.npy"))
            results[ablation_key] = ablated_store
            continue

        if ablation_key != "resid_post":
            raise ValueError("For this experiment, use ablation_keys=['resid_post'].")

        ablated_store = np.full((L + 1, S, T), np.nan, dtype=np.float32)

        # input embedding patch
        if 0 in resid_post_layers_to_run:
            for t in token_positions:
                handle = llm.layers[0].register_forward_pre_hook(
                    lambda m, inp, t=t: resample_token_inp(inp, t, source_cache["embed"][t, :])
                )
                ablated_store[0, t, :] = run_forward(target_inputs)
                handle.remove()

        # optional deeper resid_post patching (not used here, but kept)
        for l in tqdm(range(L)):
            blk = llm.layers[l]
            if (l + 1) not in resid_post_layers_to_run:
                continue
            for t in token_positions:
                handle = blk.register_forward_hook(
                    lambda m, inp, out, t=t, l=l:
                    resample_token_out(out, t, source_cache["resid_post"][l + 1, t, :])
                )
                ablated_store[l + 1, t, :] = run_forward(target_inputs)
                handle.remove()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{ablation_key}.npy"), ablated_store)

        results[ablation_key] = ablated_store

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--base_dir", type=str, default=PROJECT_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pre_img_tokens", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading/inference; replot from tables/<model>/records.json (CPU-friendly).")
    args = parser.parse_args()

    model_id = args.model
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    duck_rabbit_dir = os.path.join(args.base_dir, "data", "duck_rabbit")
    boundaries_path = os.path.join(duck_rabbit_dir, "boundaries.json")

    # changed filename
    results_dir = os.path.join(args.base_dir, "outputs", "exclusivity_patching")
    os.makedirs(results_dir, exist_ok=True)

    image_token_count = (args.image_size // args.patch_size) ** 2

    with open(boundaries_path, "r", encoding="utf-8") as f:
        boundaries = json.load(f)

    vlm_dict = {
        "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
        "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
        "llava-v1.6-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava-v1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llama3-llava-next-8b": "llava-hf/llama3-llava-next-8b-hf",
    }

    and_token_by_model = {
        "llava-1.5-7b": "▁and",
        "llava-1.5-13b": "▁and",
        "llava-v1.6-vicuna-7b": "▁and",
        "llava-v1.6-mistral-7b": "▁and",
        "llama3-llava-next-8b": "Ġand",
    }

    count_prompt_text = "How many objects are in the image? Answer with a number only."
    # Readout position for the count query = the model's own first answer step.
    # For the SentencePiece models the natural answer is "<space>1", which the
    # string prefix " " reproduces exactly (digit mass >= 0.997 and beam
    # search returns "1"/"2" there). Llama-3 answers "\n1"
    # after the template's own "\n\n", which only a token-level append can
    # reproduce (see append_token_ids in model_resampling_ablation).
    count_prefix = None if model_id == "llama3-llava-next-8b" else " "

    punct_prompt_text = "List every animal in the image, each in one word."
    punct_prefix = "I see a bird"

    print(f"Running {model_id}...")
    bird_tok, rabb_tok = get_duck_rabbit_tokens(model_id)
    and_tok = and_token_by_model[model_id]
    if args.plot_only:
        processor = model = prompt_format = None
        count_tokens = punct_tokens = count_token_to_idx = punct_token_to_idx = None
        print("[PLOT-ONLY] skipping model load; replotting from tables/records.json")
    else:
        processor, model, prompt_format = load_vlm(vlm_dict[model_id], return_prompt_format=True)
        model.to(device)
        model.eval()

        tokenizer = processor.tokenizer

        count_tokens = {
            "1": token_id_or_warn(tokenizer, "1"),
            "2": token_id_or_warn(tokenizer, "2"),
        }
        count_token_to_idx = {k: i for i, k in enumerate(count_tokens.keys())}

        # swapped order conceptually: positive = and, negative = .
        punct_tokens = {
            and_tok: token_id_or_warn(tokenizer, and_tok),
            ".": token_id_or_warn(tokenizer, "."),
        }
        punct_token_to_idx = {k: i for i, k in enumerate(punct_tokens.keys())}

    all_records = []
    # Expanded pipeline-generated set, deterministic order (va_set_manifest.json);
    # partial runs therefore cover a well-defined "first N" subset.
    with open(os.path.join(DATA_DIR, "duck_rabbit", "va_set_manifest.json")) as f:
        va_names = [f"va_s{seed:03d}" for seed in json.load(f)["seeds"]]

    # boundaries written using this prompt/prefix
    boundary_prompt_slug = slugify("List every animal in the image, each in one word." + "I see a")

    for va_name in tqdm([] if args.plot_only else va_names, desc="va images"):
        boundary = get_boundary(boundaries, model_id, va_name, boundary_prompt_slug)
        if boundary is None:
            print(f"[SKIP] No boundary for {va_name}")
            continue

        suffix = va_name.split("_")[-1]
        split_name = f"split_{suffix}"

        va_path = os.path.join(duck_rabbit_dir, f"{va_name}.png")
        split_path = os.path.join(duck_rabbit_dir, f"{split_name}.png")

        if not os.path.exists(va_path):
            print(f"[SKIP] Missing target image: {va_path}")
            continue
        if not os.path.exists(split_path):
            print(f"[SKIP] Missing source image: {split_path}")
            continue

        # target image = boundary-centered va image (in memory only)
        target_dataset = RotatedRGBAImageDataset(va_path, [int(boundary)])
        target_image = target_dataset[0]

        # source image = corresponding split image
        source_dataset = RotatedRGBAImageDataset(split_path, [0])
        source_image = source_dataset[0]

        count_prompt = build_prompt(count_prompt_text, count_prefix, prompt_format)
        count_append_ids = ([processor.tokenizer.convert_tokens_to_ids("Ċ")]
                            if model_id == "llama3-llava-next-8b" else None)
        punct_prompt = build_prompt(punct_prompt_text, punct_prefix, prompt_format)

        # style labels from source split image
        style_cache_dir = os.path.join(results_dir, "style_cache", model_id, va_name)
        style_info = compute_source_patch_styles(
            model=model,
            processor=processor,
            source_image=source_image,
            source_prompt=build_prompt(punct_prompt_text, "I see a", prompt_format),
            device=device,
            bird_tok=bird_tok,
            rabb_tok=rabb_tok,
            pre_img_tokens=args.pre_img_tokens,
            image_token_count=image_token_count,
            cache_save_dir=style_cache_dir,
            threshold=0.1,
        )

        img_positions = style_info["img_positions"]

        # x-axis: count
        count_save_dir = os.path.join(results_dir, "ablation", model_id, va_name, "count")
        count_results = model_resampling_ablation(
            model=model,
            processor=processor,
            source_image=source_image,
            target_image=target_image,
            source_prompt=count_prompt,
            target_prompt=count_prompt,
            tokens_of_interest=count_tokens,
            device=device,
            save_dir=count_save_dir,
            ablation_keys=["resid_post"],
            positions_to_ablate=img_positions,
            resid_post_layers_to_run=[0],
            append_token_ids=count_append_ids,
        )

        count_original = count_results["original"]
        count_patched = count_results["resid_post"][0]
        count_base_diff = logit_diff_from_vec(count_original, count_token_to_idx, "2", "1")

        # y-axis: and - .
        punct_save_dir = os.path.join(results_dir, "ablation", model_id, va_name, "punct")
        punct_results = model_resampling_ablation(
            model=model,
            processor=processor,
            source_image=source_image,
            target_image=target_image,
            source_prompt=punct_prompt,
            target_prompt=punct_prompt,
            tokens_of_interest=punct_tokens,
            device=device,
            save_dir=punct_save_dir,
            ablation_keys=["resid_post"],
            positions_to_ablate=img_positions,
            resid_post_layers_to_run=[0],
        )

        punct_original = punct_results["original"]
        punct_patched = punct_results["resid_post"][0]
        punct_base_diff = logit_diff_from_vec(punct_original, punct_token_to_idx, and_tok, ".")

        for local_idx, seq_pos in enumerate(img_positions):
            count_patch_diff = logit_diff_from_vec(count_patched[seq_pos], count_token_to_idx, "2", "1")
            punct_patch_diff = logit_diff_from_vec(punct_patched[seq_pos], punct_token_to_idx, and_tok, ".")

            x = float(count_patch_diff - count_base_diff)
            y = float(punct_patch_diff - punct_base_diff)

            all_records.append({
                "va_name": va_name,
                "split_name": split_name,
                "patch_idx": int(local_idx),
                "seq_pos": int(seq_pos),
                "x": x,
                "y": y,
                "color_label": style_info["color_labels"][local_idx],
                "marker_label": style_info["marker_labels"][local_idx],
            })

        # helps fragmentation a bit across images
        torch.cuda.empty_cache()

    table_dir = os.path.join(results_dir, "tables", model_id)
    os.makedirs(table_dir, exist_ok=True)
    records_path = os.path.join(table_dir, "records.json")
    if args.plot_only:
        if not os.path.exists(records_path):
            raise SystemExit(f"[PLOT-ONLY] missing cache: {records_path}")
        with open(records_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)
        print(f"[PLOT-ONLY] loaded {len(all_records)} records from {records_path}")
    else:
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

    plot_dir = os.path.join(results_dir, "plots", model_id)
    os.makedirs(plot_dir, exist_ok=True)
    plot_cross_task_scatter(
        all_records,
        save_path=os.path.join(plot_dir, "count_vs_punct_scatter.png"),
        #     f"{model_id}\n"
        #     "Per-image-token embedding patch: split_i → centered va_i\n"
        #     'x: Δ(2-1), y: Δ(and-.) after prefix "I see a bird"'
    )

    del model, processor
    torch.cuda.empty_cache()