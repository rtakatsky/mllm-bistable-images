import glob
import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from typing import List, Dict, Optional, Union, Literal, Any, Tuple
import random
import numpy as np
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.getLogger("transformers").setLevel(logging.ERROR)


# -------------------------
# PATCH: original extractor with anchor_angle support
# -------------------------
def extract_topk_from_file(
    input_path: str,
    k: int,
    extra_tokens: List[str] = [],
    anchor_angle: Optional[float] = 0,   # NEW: choose top-k from this angle
) -> List[Dict[str, Any]]:
    """
    Read JSON and:
      - choose top-k tokens from anchor_angle output (or first valid output if not found)
      - collect those token probabilities for all angles in the file
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1) choose anchor output for top-k token selection
    anchor_output = None
    if anchor_angle is not None:
        for item in data:
            if item.get("angle") == anchor_angle and item.get("output"):
                anchor_output = item["output"][0]
                break

    # fallback: first valid output
    if anchor_output is None:
        for item in data:
            if item.get("output"):
                anchor_output = item["output"][0]
                break

    if not anchor_output:
        raise ValueError(f"No valid output found in JSON: {input_path}")

    # 1-a) sort by prob and take top-k
    zpairs = sorted(
        zip(anchor_output.get('tokens', []), anchor_output.get('probs', [])),
        key=lambda tp: tp[1],
        reverse=True
    )[:k]
    topk_tokens = [tok for tok, _ in zpairs]

    # 1-b) append extra tokens if missing
    for tok in extra_tokens:
        if tok not in topk_tokens:
            topk_tokens.append(tok)

    # 2) collect probs for all angles
    topk_data = []
    for item in data:
        angle = item.get('angle')
        outputs = item.get('output', [])
        if not outputs:
            continue

        out = outputs[0]
        prob_map = dict(zip(out.get('tokens', []), out.get('probs', [])))
        probs = [prob_map.get(tok, np.nan) for tok in topk_tokens]

        topk_data.append({
            'angle': angle,
            'tokens': topk_tokens,
            'probs': probs,
        })

    return topk_data


# -------------------------
# Helpers for cue-effect plotting
# -------------------------
def build_token_angle_map(
    topk_data: List[Dict[str, Any]],
    center_angle: float,
) -> Dict[float, Dict[str, float]]:
    """
    Convert one prob_data into:
      rel_angle -> {token: prob}
    where rel_angle = angle - center_angle
    """
    out: Dict[float, Dict[str, float]] = {}
    for item in topk_data:
        ang = float(item["angle"]) - float(center_angle)
        toks = item.get("tokens", [])
        probs = item.get("probs", [])
        out[ang] = dict(zip(toks, probs))
    return out


def _subplot_grid(n: int) -> Tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 3, 4
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols

def plot_cue_effects_single(
    cue_to_record: Dict[str, Dict[str, Any]],
    cue_type: str,
    cue_order: List[str],
    color_dict: Dict[str, str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    One figure for one image (+ one base prompt + one cue_type).
    Mean/std is taken across relative angles.

    Visual encoding:
      - color: token
      - dashed thin lines: each relative-angle curve
      - solid line + shaded area: mean ± std across angles
    """
    if not cue_to_record:
        return

    x_labels = ["no cue"] + list(cue_order)
    x = np.arange(len(x_labels))
    token_order = list(color_dict.keys())

    # cue_maps[cue_label][rel_angle][token] = prob
    cue_maps: Dict[str, Dict[float, Dict[str, float]]] = {}
    all_rel_angles = set()

    for cue_label, rec in cue_to_record.items():
        prob_data = rec["prob_data"]
        center_angle = rec["center_angle"]
        amap = build_token_angle_map(prob_data, center_angle)
        cue_maps[cue_label] = amap
        all_rel_angles.update(amap.keys())

    if not all_rel_angles:
        return

    rel_angles = sorted(all_rel_angles)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for tok in token_order:
        # Y shape = [n_angles, n_cues]
        Y = np.full((len(rel_angles), len(x_labels)), np.nan, dtype=float)

        for i, rel_ang in enumerate(rel_angles):
            for j, cue in enumerate(x_labels):
                p = cue_maps.get(cue, {}).get(rel_ang, {}).get(tok, np.nan)
                Y[i, j] = p

        if np.all(np.isnan(Y)):
            continue

        color = color_dict.get(tok, None)

        # Dashed individual angle curves
        for i in range(Y.shape[0]):
            yi = Y[i]
            if np.all(np.isnan(yi)):
                continue
            ax.plot(
                x, yi,
                linestyle="--",
                linewidth=0.9,
                alpha=0.28,
                color=color,
                label="_nolegend_",
            )

        # Mean/std across angles
        counts = np.sum(np.isfinite(Y), axis=0)
        sums = np.nansum(Y, axis=0)
        mean = np.divide(
            sums, counts,
            out=np.full(len(x_labels), np.nan, dtype=float),
            where=counts > 0
        )

        diffs = np.where(np.isfinite(Y), Y - mean[None, :], 0.0)
        var = np.divide(
            np.sum(diffs ** 2, axis=0),
            counts,
            out=np.full(len(x_labels), np.nan, dtype=float),
            where=counts > 0
        )
        std = np.sqrt(var)

        valid_std = counts >= 2
        if np.any(valid_std):
            lower = np.clip(mean - std, 0, 1)
            upper = np.clip(mean + std, 0, 1)
            ax.fill_between(
                x, lower, upper,
                where=valid_std,
                alpha=0.15,
                color=color,
                linewidth=0,
            )

        ax.plot(
            x, mean,
            marker="o",
            linewidth=2.0,
            color=color,
            label=tok,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{cue_type} cue")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq_h, uniq_l = [], []
    seen = set()
    for h, l in zip(handles, labels):
        if l == "_nolegend_" or l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if uniq_h:
        ax.legend(
            uniq_h, uniq_l,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)


def plot_cue_effects_va_all(
    va_records: List[Dict[str, Dict[str, Any]]],
    cue_type: str,
    cue_order: List[str],
    color_dict: Dict[str, str],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Aggregate across va_i images for one base prompt + one cue_type.
    Mean/std is taken across ALL samples = (image, relative-angle).

    Visual encoding:
      - color: token
      - dashed thin lines: each sample curve (one per image×angle)
      - solid line + shaded area: mean ± std across all samples
    """
    if len(va_records) == 0:
        return

    x_labels = ["no cue"] + list(cue_order)
    x = np.arange(len(x_labels))
    token_order = list(color_dict.keys())

    # Build all sample maps, each sample is one (va_image, rel_angle)
    # sample_maps: List[ Dict[cue_label -> Dict[token -> prob]] ]
    sample_maps = []

    for cue_to_record in va_records:
        # cue_maps[cue_label][rel_angle][token] = prob
        cue_maps: Dict[str, Dict[float, Dict[str, float]]] = {}
        all_rel_angles = set()

        for cue_label, rec in cue_to_record.items():
            prob_data = rec["prob_data"]
            center_angle = rec["center_angle"]
            amap = build_token_angle_map(prob_data, center_angle)
            cue_maps[cue_label] = amap
            all_rel_angles.update(amap.keys())

        for rel_ang in sorted(all_rel_angles):
            one_sample = {}
            for cue_label in cue_maps.keys():
                one_sample[cue_label] = cue_maps.get(cue_label, {}).get(rel_ang, {})
            sample_maps.append(one_sample)

    if len(sample_maps) == 0:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for tok in token_order:
        # Y shape = [n_samples, n_cues]
        Y = np.full((len(sample_maps), len(x_labels)), np.nan, dtype=float)

        for i, one_sample in enumerate(sample_maps):
            for j, cue in enumerate(x_labels):
                p = one_sample.get(cue, {}).get(tok, np.nan)
                Y[i, j] = p

        if np.all(np.isnan(Y)):
            continue

        color = color_dict.get(tok, None)

        counts = np.sum(np.isfinite(Y), axis=0)
        sums = np.nansum(Y, axis=0)
        mean = np.divide(
            sums, counts,
            out=np.full(len(x_labels), np.nan, dtype=float),
            where=counts > 0
        )

        diffs = np.where(np.isfinite(Y), Y - mean[None, :], 0.0)
        var = np.divide(
            np.sum(diffs ** 2, axis=0),
            counts,
            out=np.full(len(x_labels), np.nan, dtype=float),
            where=counts > 0
        )
        std = np.sqrt(var)

        valid_std = counts >= 2
        if np.any(valid_std):
            lower = np.clip(mean - std, 0, 1)
            upper = np.clip(mean + std, 0, 1)
            ax.fill_between(
                x, lower, upper,
                where=valid_std,
                alpha=0.15,
                color=color,
                linewidth=0,
            )

        ax.plot(
            x, mean,
            marker="o",
            linewidth=2.2,
            color=color,
            label=tok,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    if cue_type == "prefix":
        ax.set_xlabel("It starts with '{cue}'.")
    elif cue_type == "semantic":
        ax.set_xlabel("It's {cue}.")
    elif cue_type == "prefix_negation":
        ax.set_xlabel("It doesn't start with '{cue}'.")
    elif cue_type == "semantic_negation":
        ax.set_xlabel("It's not {cue}.")

    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq_h, uniq_l = [], []
    seen = set()
    for h, l in zip(handles, labels):
        if l == "_nolegend_" or l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if uniq_h:
        ax.legend(
            uniq_h, uniq_l,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)
# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading and inference; only regenerate plots from cached results (CPU-friendly).")
    args = parser.parse_args()
    model_id = args.model
    offset = args.offset

    bistable_image_dir = os.path.join(DATA_DIR, 'duck_rabbit')
    bistable_image_paths = [os.path.join(bistable_image_dir, f"{image_name}.png") for image_name in ["harper"]]
    # The Visual Anagram set (see data/duck_rabbit/va_set_manifest.json).
    bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "va_s*.png")))
    # split_s*, duck_s*, rabbit_s* are included for all seeds; the loop filters
    # out ones whose va_s* has no boundary for the current prompt.
    # duck_s* / rabbit_s* are non-ambiguous single-object controls derived from
    # split_s* (make_singles_from_splits.py: other object's alpha zeroed,
    # same canvas). They probe whether prefix cues can override clear visual
    # evidence.
    bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "split_s*.png")))
    bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "duck_s*.png")))
    bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "rabbit_s*.png")))

    results_dir = os.path.join(OUTPUTS_DIR, 'top_down_cues')
    os.makedirs(results_dir, exist_ok=True)

    prompts_neutral = [
        ["List every animal in the image, each in one word.", "I see a"],
        # ["List every animal in the image, each in one word.", "There's a"],
    ]

    top_down_cues = {
        "prefix": ["b", "bi", "bu", "r", "d", "x"],
        "semantic": ["avian", "an egg-layer", "nesting", "singing", "terrestrial", "viviparous", "herbivorous", "Easter"],
        "prefix_negation": ["b", "bi", "bu", "r", "d", "x"],
        "semantic_negation": ["avian", "an egg-layer", "nesting", "singing", "terrestrial", "viviparous", "herbivorous", "Easter"]
    }

    # Build prompt specs with cue metadata + base prompt_slug
    prompt_specs = []
    for base_prompt, base_answer_prefix in prompts_neutral:
        base_prompt_slug = slugify(base_prompt + base_answer_prefix)

        # neutral
        prompt_specs.append({
            "prompt": base_prompt,
            "prompt_prefix": base_answer_prefix,
            "cue_type": "none",
            "cue_value": "no cue",
            "base_prompt_slug": base_prompt_slug,
            "prompt_slug": slugify(base_prompt + base_answer_prefix),
        })

        # prefix cues
        for prefix in top_down_cues["prefix"]:
            p = base_prompt + f" It starts with '{prefix}'."
            prompt_specs.append({
                "prompt": p,
                "prompt_prefix": base_answer_prefix,
                "cue_type": "prefix",
                "cue_value": prefix,
                "base_prompt_slug": base_prompt_slug,
                "prompt_slug": slugify(p + base_answer_prefix),
            })

        # semantic cues
        for semantic in top_down_cues["semantic"]:
            p = base_prompt + f" It's {semantic}."
            prompt_specs.append({
                "prompt": p,
                "prompt_prefix": base_answer_prefix,
                "cue_type": "semantic",
                "cue_value": semantic,
                "base_prompt_slug": base_prompt_slug,
                "prompt_slug": slugify(p + base_answer_prefix),
            })
        
        # negation cues
        for negation in top_down_cues["prefix_negation"]:
            p = base_prompt + f" It doesn't start with '{negation}'."
            prompt_specs.append({
                "prompt": p,
                "prompt_prefix": base_answer_prefix,
                "cue_type": "prefix_negation",
                "cue_value": negation,
                "base_prompt_slug": base_prompt_slug,
                "prompt_slug": slugify(p + base_answer_prefix),
            })

        for negation in top_down_cues["semantic_negation"]:
            p = base_prompt + f" It's not {negation}."
            prompt_specs.append({
                "prompt": p,
                "prompt_prefix": base_answer_prefix,
                "cue_type": "semantic_negation",
                "cue_value": negation,
                "base_prompt_slug": base_prompt_slug,
                "prompt_slug": slugify(p + base_answer_prefix),
            })
    color_dict = {
        "▁du": "blue",
        # "▁Duck": "blue",
        "▁duck": "blue",
        "Ġduck": "blue",
        "▁bird": "skyblue",
        "Ġbird": "skyblue",
        "▁rabb": "orange",
        # "▁Rab": "orange",
        "▁rab": "orange",
        "Ġrabbit": "orange",
        "▁b": "salmon",
        # "▁Bun": "salmon",
        "Ġbunny": "salmon",
    }

    sample_n = 1
    temperature = 1.0
    max_new_tokens = 1
    seed = 42
    batch_size = 2
    use_logit = True
    top_k = 100

    # tasks
    tasks = [
        (image_path, spec)
        for image_path in bistable_image_paths
        for spec in prompt_specs
    ]

    os.makedirs(os.path.join(results_dir, "text"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    print(f"Running {model_id}...")
    if args.plot_only:
        processor = model = prompt_format = None
        print("[PLOT-ONLY] skipping model load; replotting cached results")
    else:
        processor, model, prompt_format = load_vlm(VLM_DICT[model_id], return_prompt_format=True)

    with open(os.path.join(bistable_image_dir, "boundaries.json"), "r") as f:
        boundaries = json.load(f)

    # Records for plotting:
    cue_records: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    cue_orders = {
        "prefix": top_down_cues["prefix"],
        "semantic": top_down_cues["semantic"],
        "prefix_negation": top_down_cues["prefix_negation"],
        "semantic_negation": top_down_cues["semantic_negation"],
    }

    for image_path, spec in tqdm(tasks, desc='All tasks'):
        image_name = os.path.basename(image_path).split('.')[0]

        prompt = spec["prompt"]
        prompt_prefix = spec["prompt_prefix"]
        prompt_slug = spec["prompt_slug"]
        base_prompt_slug = spec["base_prompt_slug"]
        cue_type = spec["cue_type"]
        cue_value = spec["cue_value"]

        os.makedirs(os.path.join(results_dir, "text", model_id, image_name), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots", model_id, image_name), exist_ok=True)
        text_path = os.path.join(results_dir, "text", model_id, image_name, f"{prompt_slug}.json")

        # Skip split_i / duck_i / rabbit_i if the corresponding va_i has no
        # boundary for this base prompt slug.
        if (image_name.startswith("split_")
                or image_name.startswith("duck_")
                or image_name.startswith("rabbit_")):
            suffix = image_name.split("_")[-1]
            if get_boundary(boundaries, model_id, f"va_{suffix}", base_prompt_slug) is None:
                continue

        # Angle selection (uses center ± offset for va_i)
        if image_name.startswith("va_"):
            center_angle = get_boundary(
                boundaries=boundaries,
                model_name=model_id,
                image_name=image_name,
                prompt_slug=base_prompt_slug,  # use base prompt+prefix slug
            )
            if center_angle is None:
                print(f"[WARN] {image_name} | {base_prompt_slug}: boundary not found, skipping")
                continue
            angles = [i for i in range(int(center_angle) - offset, int(center_angle) + offset + 1)]
        else:
            center_angle = 0
            angles = [i for i in range(-offset, offset+1)]

        dataset = RotatedRGBAImageDataset(image_path, angles)

        if os.path.exists(text_path) and (args.plot_only or not args.overwrite):
            print(f"[SKIP] {model_id} results already exist: {text_path}")
        elif args.plot_only:
            print(f"[PLOT-ONLY SKIP] missing cache: {text_path}")
            continue
        else:
            results = run_vlm_sampling(
                model, processor, dataset, prompt,
                sample_n=sample_n, temperature=temperature, max_new_tokens=max_new_tokens,
                seed=seed, batch_size=batch_size,
                use_logit=use_logit, top_k=top_k,
                prompt_format=prompt_format,
                prompt_prefix=prompt_prefix,
                extra_tokens=color_dict.keys(),
            )

            with open(text_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        # Use original extractor, but anchor top-k at center_angle for va_i
        prob_data = extract_topk_from_file(
            text_path,
            k=5,
            extra_tokens=list(color_dict.keys()),
            anchor_angle=center_angle,
        )

        # Store records for final plotting
        # Put neutral into both cue plots so "no cue" appears in both
        if cue_type == "none":
            cue_records[(image_name, base_prompt_slug, "prefix")]["no cue"] = {
                "prob_data": prob_data,
                "center_angle": center_angle,
            }
            cue_records[(image_name, base_prompt_slug, "semantic")]["no cue"] = {
                "prob_data": prob_data,
                "center_angle": center_angle,
            }
            cue_records[(image_name, base_prompt_slug, "prefix_negation")]["no cue"] = {
                "prob_data": prob_data,
                "center_angle": center_angle,
            }
            cue_records[(image_name, base_prompt_slug, "semantic_negation")]["no cue"] = {
                "prob_data": prob_data,
                "center_angle": center_angle,
            }
        else:
            cue_records[(image_name, base_prompt_slug, cue_type)][cue_value] = {
                "prob_data": prob_data,
                "center_angle": center_angle,
            }

    del model, processor
    torch.cuda.empty_cache()

    # -------------------------
    # Plot top-down cue effects
    # -------------------------

    # Per-image plots
    for (image_name, base_prompt_slug, cue_type), cue_to_record in cue_records.items():
        if cue_type not in ("prefix", "semantic", "prefix_negation", "semantic_negation"):
            continue

        # Save directly under results_dir/plots/.../image_name using "original style" prompt_slug-like naming
        save_path = os.path.join(
            results_dir, "plots", model_id, image_name, f"{base_prompt_slug}_{cue_type}.png"
        )

        plot_cue_effects_single(
            cue_to_record=cue_to_record,
            cue_type=cue_type,
            cue_order=cue_orders[cue_type],
            color_dict=color_dict,
            title=f"{model_id} | {image_name} | {cue_type}",
            save_path=save_path,
        )

    # va_all aggregate plots
    va_grouped: Dict[Tuple[str, str], List[Dict[str, Dict[str, Any]]]] = defaultdict(list)
    # split_all (non-ambiguous control) aggregate plots
    split_grouped: Dict[Tuple[str, str], List[Dict[str, Dict[str, Any]]]] = defaultdict(list)
    # duck_all / rabbit_all aggregate plots (single-animal masked controls).
    # These probe whether prefix cues can override unambiguous visual evidence.
    duck_grouped:   Dict[Tuple[str, str], List[Dict[str, Dict[str, Any]]]] = defaultdict(list)
    rabbit_grouped: Dict[Tuple[str, str], List[Dict[str, Dict[str, Any]]]] = defaultdict(list)
    for (image_name, base_prompt_slug, cue_type), cue_to_record in cue_records.items():
        if cue_type not in ("prefix", "semantic", "prefix_negation", "semantic_negation"):
            continue
        if image_name.startswith("va_"):
            va_grouped[(base_prompt_slug, cue_type)].append(cue_to_record)
        elif image_name.startswith("split_"):
            split_grouped[(base_prompt_slug, cue_type)].append(cue_to_record)
        elif image_name.startswith("duck_"):
            duck_grouped[(base_prompt_slug, cue_type)].append(cue_to_record)
        elif image_name.startswith("rabbit_"):
            rabbit_grouped[(base_prompt_slug, cue_type)].append(cue_to_record)

    for (base_prompt_slug, cue_type), va_records in va_grouped.items():
        if len(va_records) == 0:
            continue

        save_path = os.path.join(
            results_dir, "plots", model_id, "va_all", f"{base_prompt_slug}_{cue_type}.png"
        )

        plot_cue_effects_va_all(
            va_records=va_records,
            cue_type=cue_type,
            cue_order=cue_orders[cue_type],
            color_dict=color_dict,
            save_path=save_path,
        )

    for (base_prompt_slug, cue_type), split_records in split_grouped.items():
        if len(split_records) == 0:
            continue

        save_path = os.path.join(
            results_dir, "plots", model_id, "split_all", f"{base_prompt_slug}_{cue_type}.png"
        )

        plot_cue_effects_va_all(
            va_records=split_records,
            cue_type=cue_type,
            cue_order=cue_orders[cue_type],
            color_dict=color_dict,
            save_path=save_path,
        )

    for (base_prompt_slug, cue_type), duck_records in duck_grouped.items():
        if len(duck_records) == 0:
            continue
        save_path = os.path.join(
            results_dir, "plots", model_id, "duck_all", f"{base_prompt_slug}_{cue_type}.png"
        )
        plot_cue_effects_va_all(
            va_records=duck_records,
            cue_type=cue_type,
            cue_order=cue_orders[cue_type],
            color_dict=color_dict,
            save_path=save_path,
        )

    for (base_prompt_slug, cue_type), rabbit_records in rabbit_grouped.items():
        if len(rabbit_records) == 0:
            continue
        save_path = os.path.join(
            results_dir, "plots", model_id, "rabbit_all", f"{base_prompt_slug}_{cue_type}.png"
        )
        plot_cue_effects_va_all(
            va_records=rabbit_records,
            cue_type=cue_type,
            cue_order=cue_orders[cue_type],
            color_dict=color_dict,
            save_path=save_path,
        )