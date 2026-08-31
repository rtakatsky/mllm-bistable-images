# Rotating a duck-rabbit -- keyword-matching analysis.
# Mirrors the structure of rotation_beam.py:
# for VA images the rotation window is centered on the saved boundary
# (from boundaries.json) so curves are aligned across VA samples.

import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import argparse
import logging
from collections import Counter
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.ERROR)


def classify_sentence_keyword_matching(
    text: str,
    mapping: Dict[str, List[str]],
) -> Literal["rabbit", "duck", "both", "other"]:
    hits = {label for label, kws in mapping.items()
            if any(kw in text.lower() for kw in kws)}
    if hits == {"rabbit"}:
        return "rabbit"
    if hits == {"duck"}:
        return "duck"
    if hits == {"duck", "rabbit"}:
        return "both"
    return "other"


def process_results_json(
    input_path: str,
    output_path: str,
    mapping: Dict[str, List[str]],
) -> List[Dict[str, float]]:
    with open(input_path, "r") as f:
        data = json.load(f)
    prob_data = []
    for item in data:
        angle = item["angle"]
        outputs = item["output"]
        labels = [classify_sentence_keyword_matching(t, mapping) for t in outputs]
        counts = Counter(labels)
        total = len(outputs)
        probs = {label: counts.get(label, 0) / total for label in ["duck", "rabbit", "both", "other"]}
        prob_data.append({"angle": angle, **probs})
    with open(output_path, "w") as f:
        json.dump(prob_data, f, ensure_ascii=False, indent=2)
    return prob_data


def plot_probabilities_aggregate(
    prob_data_list: List[List[Dict[str, float]]],
    center_angles: List[int],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot mean duck/rabbit/both/other probabilities across multiple runs,
    aligned by relative angle (angle - center_angle)."""
    from collections import defaultdict

    acc: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"duck": [], "rabbit": [], "both": [], "other": []}
    )
    for prob_data, center in zip(prob_data_list, center_angles):
        for item in prob_data:
            rel = int(item["angle"] - center)
            for label in ("duck", "rabbit", "both", "other"):
                acc[rel][label].append(float(item[label]))

    if not acc:
        return
    rel_sorted = sorted(acc.keys())
    means = {
        label: [float(np.mean(acc[r][label])) for r in rel_sorted]
        for label in ("duck", "rabbit", "both", "other")
    }

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    display = {"duck": "duck-only", "rabbit": "rabbit-only", "both": "both", "other": "other"}
    for label in ("duck", "rabbit", "both", "other"):
        ax.plot(rel_sorted, means[label], label=display[label])
    ax.set_xlabel("Angle (relative to center)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close()


def plot_probabilities(
    prob_data: List[Dict[str, float]],
    image_path: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    center_angle: int = 0,
) -> None:
    """Plot duck/rabbit/both/other probabilities vs. (angle - center_angle)."""
    sorted_data = sorted(prob_data, key=lambda x: x["angle"])
    angles = [item["angle"] - center_angle for item in sorted_data]
    duck   = [item["duck"]   for item in sorted_data]
    rabbit = [item["rabbit"] for item in sorted_data]
    both   = [item["both"]   for item in sorted_data]
    other  = [item["other"]  for item in sorted_data]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(angles, duck,   label="duck-only")
    ax.plot(angles, rabbit, label="rabbit-only")
    ax.plot(angles, both,   label="both")
    ax.plot(angles, other,  label="other")
    ax.set_xlabel("Angle (relative to center)" if center_angle else "Angle")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True)


    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--image_name", type=str, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    # Within-model parallelism: task i of n processes bistable_image_paths[i::n].
    # Per-image outputs are independent; run a final pass without chunking (all
    # tasks skip) to rebuild the va_all/split_all aggregate plots.
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading and inference; only regenerate plots from cached results (CPU-friendly).")
    parser.add_argument("--sample-n-generated", type=int, default=8,
                        help="Samples per angle for the generated (va_s*/split_s*) stimuli. The default 8 is "
                             "what the original 10-model runs used; aggregates aritmetically average over ~40 "
                             "valid images per model, so a smaller value still gives a large effective n per "
                             "angle and cuts runtime proportionally. harper keeps sample_n_harper (single image).")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    args = parser.parse_args()
    model_id = args.model

    bistable_image_dir = os.path.join(DATA_DIR, "duck_rabbit")

    results_dir = os.path.join(OUTPUTS_DIR, "rotation_freeform")
    os.makedirs(results_dir, exist_ok=True)

    angle_lim = 60
    rel_angles = np.arange(-angle_lim, angle_lim + 1, 1).tolist()

    prompts = [
        # General
        ["Describe this image.", ""],
        # CoT
        ["Describe this image. Think step by step.", ""],
    ]

    # Boundary reference prompt: VA images are rotated relative to the center
    # angle saved under this slug in boundaries.json, regardless of the actual
    # prompt being evaluated here.
    boundary_ref_prompt = "List every animal in the image, each in one word."
    boundary_ref_prefix = "I see a"
    boundary_ref_slug = slugify(boundary_ref_prompt + boundary_ref_prefix)

    boundaries_path = os.path.join(bistable_image_dir, "boundaries.json")
    if os.path.exists(boundaries_path):
        with open(boundaries_path, "r", encoding="utf-8") as f:
            boundaries = json.load(f)
    else:
        boundaries = {}
        print(f"[WARN] boundaries file not found: {boundaries_path}")

    # The Visual Anagram set (see data/duck_rabbit/va_set_manifest.json).
    # Valid va_s* = those with a saved boundary for the reference slug.
    # Their corresponding split images are included too (at center_angle=0).
    va_names = sorted(
        os.path.basename(p)[:-len(".png")]
        for p in glob.glob(os.path.join(bistable_image_dir, "va_s*.png"))
    )
    valid_va_names = [
        n for n in va_names
        if get_boundary(boundaries, model_id, n, boundary_ref_slug) is not None
    ]
    print(f"[INFO] valid va names for {model_id}: {valid_va_names}")

    if args.image_name is None:
        bistable_image_paths = [os.path.join(bistable_image_dir, f"{n}.png")
                                for n in ["harper"]]
        bistable_image_paths += [os.path.join(bistable_image_dir, f"{n}.png") for n in va_names]
        bistable_image_paths += [os.path.join(bistable_image_dir, n.replace("va_", "split_", 1) + ".png")
                                 for n in valid_va_names]
    else:
        bistable_image_paths = [os.path.join(bistable_image_dir, f"{args.image_name}.png")]

    if args.num_chunks > 1:
        bistable_image_paths = bistable_image_paths[args.chunk_idx::args.num_chunks]
        print(f"[INFO] chunk {args.chunk_idx}/{args.num_chunks}: {len(bistable_image_paths)} images")

    mapping = {"duck": ["bird", "duck", "goose"], "rabbit": ["rabbit", "bunny"]}

    # harper keeps the original 64 samples/angle (single image, headline panel);
    # va_s*/split_s* use 8 — the va_all/split_all aggregates now pool ~40 valid
    # images per model, so per-angle precision comes from the stimulus dimension
    # (8 x ~40 = ~320 samples per aggregate point).
    sample_n_harper = 64
    sample_n_generated = args.sample_n_generated
    temperature = 1.0
    max_new_tokens = 128
    seed = 42
    batch_size = 2
    use_logit = False

    tasks = [(image_path, text_prompt)
             for image_path in bistable_image_paths
             for text_prompt in prompts]

    print(f"Running {model_id}...")
    if args.plot_only:
        processor = model = prompt_format = None
        print('[PLOT-ONLY] skipping model load; will only replot cached results')
    else:
        processor, model, prompt_format = load_vlm(VLM_DICT[model_id], return_prompt_format=True)

    # va_all and split_all aggregates keyed by prompt_slug.
    # Each entry holds prob_data lists + center angles for downstream averaging.
    va_aggregate: Dict[str, Dict[str, list]] = {}
    split_aggregate: Dict[str, Dict[str, list]] = {}

    for image_path, text_prompt in tqdm(tasks, desc="All tasks"):
        image_name = os.path.basename(image_path).split(".")[0]
        prompt, prompt_prefix = text_prompt[0], text_prompt[1]
        prompt_slug = slugify(prompt + prompt_prefix)

        text_path = os.path.join(results_dir, "text",  model_id, image_name, f"{prompt_slug}.json")
        prob_path = os.path.join(results_dir, "probs", model_id, image_name, f"{prompt_slug}.json")
        plot_path = os.path.join(results_dir, "plots", model_id, image_name, f"{prompt_slug}.png")
        for p in (text_path, prob_path, plot_path):
            os.makedirs(os.path.dirname(p), exist_ok=True)

        # Center angle: VA images use the saved boundary; others stay at 0.
        if image_name.startswith("va_"):
            center_angle = get_boundary(
                boundaries=boundaries,
                model_name=model_id,
                image_name=image_name,
                prompt_slug=boundary_ref_slug,
            )
            if center_angle is None:
                print(f"[SKIP] No boundary for {model_id} | {image_name} | {boundary_ref_slug}")
                continue
        else:
            center_angle = 0

        angles = [center_angle + da for da in rel_angles]

        if (not args.overwrite) and os.path.exists(text_path):
            print(f"[SKIP] {model_id} results already exist: {text_path}")
        elif args.plot_only:
            print(f'[PLOT-ONLY SKIP] missing cache: {text_path}')
            continue
        else:
            dataset = RotatedRGBAImageDataset(image_path, angles)
            sample_n = sample_n_harper if image_name == "harper" else sample_n_generated
            results = run_vlm_sampling(
                model, processor, dataset, prompt,
                sample_n=sample_n, temperature=temperature, max_new_tokens=max_new_tokens,
                seed=seed, batch_size=batch_size,
                use_logit=use_logit,
                prompt_format=prompt_format,
                prompt_prefix=prompt_prefix,
            )
            with open(text_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        prob_data = process_results_json(text_path, prob_path, mapping)
        plot_probabilities(
            prob_data,
            image_path,
            title=f"{model_id}: {prompt} prefix: {prompt_prefix}",
            save_path=plot_path,
            center_angle=center_angle,
        )

        # Aggregate for va_all / split_all plots (rotation_beam/rotation_first_token style).
        if image_name.startswith("va_"):
            va_aggregate.setdefault(prompt_slug, {"runs": [], "centers": []})
            va_aggregate[prompt_slug]["runs"].append(prob_data)
            va_aggregate[prompt_slug]["centers"].append(center_angle)
        elif image_name.startswith("split_"):
            split_aggregate.setdefault(prompt_slug, {"runs": [], "centers": []})
            split_aggregate[prompt_slug]["runs"].append(prob_data)
            split_aggregate[prompt_slug]["centers"].append(0)

    # va_all and split_all aggregate plots.
    for prompt_slug, pack in va_aggregate.items():
        runs = pack["runs"]
        centers = pack["centers"]
        if not runs:
            continue
        save_path = os.path.join(results_dir, "plots", model_id, "va_all", f"{prompt_slug}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_probabilities_aggregate(runs, centers, save_path=save_path)

    for prompt_slug, pack in split_aggregate.items():
        runs = pack["runs"]
        centers = pack["centers"]
        if not runs:
            continue
        save_path = os.path.join(results_dir, "plots", model_id, "split_all", f"{prompt_slug}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_probabilities_aggregate(runs, centers, save_path=save_path)

    del model, processor
    torch.cuda.empty_cache()
