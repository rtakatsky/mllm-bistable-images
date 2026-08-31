import glob
import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

from typing import List, Dict, Optional, Union, Literal
import random
import numpy as np
from tqdm import tqdm
import argparse
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import json
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.image as mpimg

def extract_topk_from_file(
    input_path: str,
    k: int,
    extra_tokens: List[str] = [],
) -> List[Dict[str, Any]]:
    """Read a results JSON, fix the top-k tokens from the angle==0 output, and
    return the probability of each of those tokens at every angle (nan when the
    token is absent).

    Args:
        input_path: path to the results JSON
        k: number of tokens to keep

    Returns:
        List of dicts with 'angle', 'tokens' (the same top-k list at every
        angle) and 'probs' (their probabilities at that angle).
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

  # 1) pick the top-k tokens from the angle==0 output
    zero_output = next(
        (item['output'][0] for item in data
         if item.get('angle') == 0 and item.get('output')),
        None
    )
    if not zero_output:
        raise ValueError("angle==0 not found in JSON")

  # 1a) sort the angle==0 tokens by probability and keep the top k
    zpairs = sorted(
        zip(zero_output['tokens'], zero_output['probs']),
        key=lambda tp: tp[1],
        reverse=True
    )[:k]
    topk_tokens = [tok for tok, _ in zpairs]

  # 1b) append the additional tokens (without duplicates)
    for tok in extra_tokens:
        if tok not in topk_tokens:
            topk_tokens.append(tok)

  # 2) collect the probabilities at every angle
    topk_data = []
    for item in data:
        angle = item.get('angle')
        outputs = item.get('output', [])
        if not outputs:
            continue

        out = outputs[0]
  # all tokens and probabilities at this angle
        prob_map = dict(zip(out.get('tokens', []),
                            out.get('probs', [])))

  # probabilities in topk_tokens order, np.nan when absent
        probs = [prob_map.get(tok, np.nan) for tok in topk_tokens]

        topk_data.append({
            'angle':  angle,
            'tokens': topk_tokens,
            'probs':  probs,
        })

    return topk_data

from typing import Sequence
def decision_boundary_from_topk(
    topk_data: List[Dict[str, Any]],
    token_a: str,
    token_b: str,
    angle_lim: Tuple[float, float] = (-45, 45),
    eps: float = 1e-6,
    min_points: int = 5,
    clamp_to_range: bool = True,
    plot: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    # reject "almost stable" cases
    min_pred_range: float = 0.15,         # predicted max(p)-min(p) within angle_lim
    min_max_slope: float = 0.002,         # max |dp/dθ| (per degree); for logistic it's |a|/4
    # fitting knobs
    slope_grid: Optional[np.ndarray] = None,  # if None, uses a decent default
    denom_quantile: float = 0.0,          # optionally ignore lowest-denom points (0.0 keeps all)
    min_count: int = 3,
    thresholds: Tuple[float, float] = (0.4, 0.6),
) -> Optional[int]:
    """
    Fit a binary psychometric curve between token_a vs token_b using topk_data and
    return the decision boundary angle (integer). Uses a logistic fit in probability
    space via grid search (robust for saturated data).

    Returns None if:
      - not enough valid points
      - curve doesn't have enough range/slope (almost stable)
      - fitting degenerates
    """
    if not topk_data:
        return None

    lo, hi = angle_lim
    if lo > hi:
        lo, hi = hi, lo

    low_thr, high_thr = thresholds

    # Collect (angle, pA, pB)
    xs, pAs, pBs = [], [], []
    for item in topk_data:
        angle = item.get("angle", None)
        toks = item.get("tokens", [])
        probs = item.get("probs", [])
        if angle is None or not toks or not probs:
            continue
        angle = float(angle)
        if not (lo <= angle <= hi):
            continue

        pmap = dict(zip(toks, probs))
        pA = pmap.get(token_a, np.nan)
        pB = pmap.get(token_b, np.nan)

        xs.append(angle)
        pAs.append(pA)
        pBs.append(pB)

    x = np.asarray(xs, dtype=float)
    pA = np.asarray(pAs, dtype=float)
    pB = np.asarray(pBs, dtype=float)

    denom = pA + pB
    mask = np.isfinite(x) & np.isfinite(pA) & np.isfinite(pB) & np.isfinite(denom) & (denom > 0)
    if mask.sum() < min_points:
        return None

    x = x[mask]
    pA = pA[mask]
    pB = pB[mask]
    denom = denom[mask]

    # Optional: drop very low-evidence points where both tokens are tiny
    if denom_quantile > 0.0:
        thr = np.quantile(denom, denom_quantile)
        keep = denom >= thr
        if keep.sum() < min_points:
            return None
        x, pA, pB, denom = x[keep], pA[keep], pB[keep], denom[keep]

    # Binary-normalized probability for A
    p = pA / denom
    p = np.clip(p, eps, 1.0 - eps)

    # --- FIT: logistic in probability space via grid search (minimal but robust) ---
    # p_fit(angle) = sigmoid(a*(angle - mu))
    if slope_grid is None:
        # slopes (per degree): small -> gradual, large -> step-like
        slope_grid = np.concatenate([
            np.linspace(0.02, 0.2, 20),
            np.linspace(0.25, 2.5, 30),
        ])

    # Candidate boundaries: integer angles in-window
    mu_grid = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1, dtype=float)

    # Weights: denom (more weight when either token is relatively likely)
    w = denom.astype(float)
    w = w / (w.sum() + 1e-12)

    best_loss = np.inf
    best_mu = None
    best_a = None

    # vectorized-ish loop: iterate mu, compute best slope
    for mu in mu_grid:
        dx = x - mu  # (N,)
        # For each slope, compute p_pred
        # shape: (S, N)
        z = slope_grid[:, None] * dx[None, :]
        p_pred = 1.0 / (1.0 + np.exp(-z))
        # weighted MSE in probability space
        loss = np.sum(w[None, :] * (p_pred - p[None, :])**2, axis=1)  # (S,)
        j = int(np.argmin(loss))
        if loss[j] < best_loss:
            best_loss = float(loss[j])
            best_loss = best_loss
            best_mu = float(mu)
            best_a = float(slope_grid[j])

    if best_mu is None or best_a is None or not np.isfinite(best_mu) or not np.isfinite(best_a):
        return None

    # Reject "almost stable everywhere" using predicted range and max slope
    grid = np.linspace(lo, hi, 400)
    p_fit = 1.0 / (1.0 + np.exp(-(best_a * (grid - best_mu))))
    pred_range = float(np.max(p_fit) - np.min(p_fit))
    max_slope = float(abs(best_a) / 4.0)  # logistic max derivative

    if pred_range < min_pred_range or max_slope < min_max_slope:
        if plot:
            _plot_fit_debug(x, p, denom, grid, p_fit, best_mu, token_a, token_b, lo, hi, title, save_path)
        return None
    
    n_low  = int(np.sum(p <= low_thr))
    n_high = int(np.sum(p >= high_thr))
    if n_low < min_count or n_high < min_count:
        # optionally still plot to debug
        if plot:
            _plot_fit_debug(x, p, denom, grid, p_fit, best_mu, token_a, token_b, lo, hi, title, save_path)
        return None


    boundary = int(round(best_mu))
    if clamp_to_range:
        boundary = int(np.clip(boundary, int(np.min(x)), int(np.max(x))))

    if plot:
        _plot_fit_debug(x, p, denom, grid, p_fit, best_mu, token_a, token_b, lo, hi, title, save_path)

    return boundary


def _plot_fit_debug(
    x: np.ndarray,
    p: np.ndarray,
    denom: np.ndarray,
    grid: np.ndarray,
    p_fit: np.ndarray,
    mu: float,
    token_a: str,
    token_b: str,
    lo: float,
    hi: float,
    title: Optional[str],
    save_path: Optional[str],
):
    order = np.argsort(x)
    xs = x[order]
    ps = p[order]
    ws = denom[order]

    fig, ax = plt.subplots(figsize=(8, 5))

    # marker size scaled by denom (optional but very informative)
    s = 20 + 180 * (ws / (np.max(ws) + 1e-12))
    ax.scatter(xs, ps, s=s, label="data (pA/(pA+pB))")

    ax.plot(grid, p_fit, label="fitted logistic (prob-space)")

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.axvline(mu, linestyle="--", linewidth=1, label=f"steepest/boundary ≈ {mu:.2f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Angle")
    ax.set_ylabel(f"P({token_a}) / (P({token_a})+P({token_b}))")
    ax.set_title(title or f"Psychometric fit: {token_a} vs {token_b}")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()
    plt.close(fig)


from pathlib import Path
def update_boundary_json(
    model_name: str,
    image_name: str,
    prompt: str,
    prompt_prefix: str,
    prompt_slug: str,
    boundary: Optional[int],
    angle_lim: Tuple[int, int] = (-45, 45),
    token_a: str = "",
    token_b: str = "",
    save_path: str = "boundaries.json",  # use a NEW file
) -> Path:
    """
    Save/update decision boundary for one (model, image, prompt_slug).

    Schema:
    {
      "<model_name>": {
        "<image_name>": {
          "<prompt_slug>": {
            "boundary": <int|null>,
            "angle_lim": [...],
            "token_a": "...",
            "token_b": "...",
            "prompt": "...",
            "prompt_prefix": "..."
          }
        }
      }
    }
    """
    data: Dict[str, Any] = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}

    if model_name not in data or not isinstance(data[model_name], dict):
        data[model_name] = {}
    if image_name not in data[model_name] or not isinstance(data[model_name][image_name], dict):
        data[model_name][image_name] = {}

    data[model_name][image_name][prompt_slug] = {
        "boundary": boundary,
        "angle_lim": list(angle_lim),
        "token_a": token_a,
        "token_b": token_b,
        "prompt": prompt,
        "prompt_prefix": prompt_prefix,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return Path(save_path)

if __name__ == "__main__":
    from utils import *
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading and inference; only regenerate plots from cached results (CPU-friendly).")
    parser.add_argument("--chunk-idx", type=int, default=None,
                        help="Inference-only chunk mode: run tasks[chunk_idx::num_chunks] (text JSONs only; "
                             "no boundaries.json writes, no split gating, aggregates are partial). "
                             "Run the script once more without chunk args afterwards to fit boundaries + aggregates.")
    parser.add_argument("--num-chunks", type=int, default=None)
    # Forced-choice variant: the "bird or rabbit?" prompt in both orders, run as a plain first-token sweep. Kept out of the
    # default run (and off boundaries.json) by writing to its own results dir and
    # running in chunk mode, which is inference-only.
    parser.add_argument("--prompts", type=str, default="default", choices=["default", "forced-choice"])
    parser.add_argument("--angles", type=str, default=None,
                        help="Comma-separated absolute angles, or 'lo:hi:step' (default: -180..179 at 1 deg)")
    parser.add_argument("--results-subdir", type=str, default=None,
                        help="Write under outputs/<subdir> instead of the default experiment dir")
    parser.add_argument("--images", type=str, default=None,
                        help="Comma-separated stimulus names to restrict the run")
    parser.add_argument("--center-on-boundary", action="store_true",
                        help="Treat --angles as RELATIVE offsets and shift them per va image by its "
                             "default-query boundary from boundaries.json (as top_down_cues does); va images "
                             "without a boundary are skipped, other images are centred on 0.")
    args = parser.parse_args()
    model_id = args.model
    chunk_mode = args.chunk_idx is not None
    if chunk_mode:
        assert args.num_chunks and 0 <= args.chunk_idx < args.num_chunks

    bistable_image_dir = os.path.join(DATA_DIR, 'duck_rabbit')
    bistable_image_paths = [os.path.join(bistable_image_dir, f"{image_name}.png") for image_name in ["harper"]]
    # The Visual Anagram set (see data/duck_rabbit/va_set_manifest.json).
    # va_* must precede split_*: split_X tasks are gated on va_X boundary validity.
    bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "va_s*.png")))
    bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "split_s*.png")))
    
    if args.images is not None:
        wanted = set(args.images.split(","))
        bistable_image_paths = [p for p in bistable_image_paths
                                if os.path.basename(p).split(".")[0] in wanted]
        missing = wanted - {os.path.basename(p).split(".")[0] for p in bistable_image_paths}
        if missing:
            raise ValueError(f"--images names not found: {sorted(missing)}")

    results_dir = os.path.join(OUTPUTS_DIR, args.results_subdir or 'rotation_first_token')
    os.makedirs(results_dir, exist_ok=True)

    boundaries = None
    if args.center_on_boundary:
        with open(os.path.join(bistable_image_dir, "boundaries.json")) as f:
            boundaries = json.load(f)
    base_prompt_slug = slugify("List every animal in the image, each in one word." + "I see a")

    if args.angles is None:
        angles = np.arange(-180, 180, 1).tolist()
    elif ":" in args.angles:
        lo, hi, step = (int(x) for x in args.angles.split(":"))
        angles = np.arange(lo, hi + step, step).tolist()
    else:
        angles = [int(x) for x in args.angles.split(",")]

    prompts_neutral = [
        ["List every animal in the image, each in one word.", "I see a"],
        ["List every object in the image, each in one word.", "I see a"],
        # ["List every animal in the image, each in one word.", "There's a"],
        # ["List every animal in the image, each in one word.", "There's a duck"],
        # ["List every animal in the image, each in one word.", "There's a rabbit"],
        # ["List every animal in the image.", "There's a"],
        # ["List every animal in the image.", "There's a duck"],
        # ["List every animal in the image.", "There's a rabbit"],
    ]

    prompts = prompts_neutral.copy()
    if args.prompts == "forced-choice":
        prompts = [
            ["Is it a bird or a rabbit?", "It is a"],
            ["Is it a rabbit or a bird?", "It is a"],
        ]

    color_dict = {
        "▁du": "blue",
        # "▁Duck": "blue",
        "▁duck": "blue", # mistral
        "Ġduck": "blue", # llama
        "▁bird": "skyblue",
        "Ġbird": "skyblue", # llama
        "▁rabb": "orange",
        # "▁Rab": "orange",
        "▁rab": "orange", # mistral
        "Ġrabbit": "orange", # llama
        "▁b": "salmon",
        # "▁Bun": "salmon",
        "Ġbunny": "salmon", # llama
        # "▁drawing": "gray",
        # "Ġdrawing": "gray", # llama
        # "▁fish": "green",
        # "Ġfish": "green", # llama
    }

    sample_n=1
    temperature=1.0
    max_new_tokens=1
    seed=42
    batch_size=2
    use_logit=True
    top_k=100

    # %%
  # Skip (prompt, model) pairs whose results already exist.
  # Build the (prompt, model) task list.
    tasks = [
        (image_path, prompt)
        for image_path in bistable_image_paths
        for prompt in prompts
        # for vlm_name, model_id in VLM.items()
    ]
    if chunk_mode:
        tasks = tasks[args.chunk_idx::args.num_chunks]
        print(f"[CHUNK {args.chunk_idx}/{args.num_chunks}] {len(tasks)} tasks (inference only)")
    os.makedirs(os.path.join(results_dir, "text"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    print(f"Running {model_id}...")
    if args.plot_only:
        processor = model = prompt_format = None
        print("[PLOT-ONLY] skipping model load; will only replot cached results")
    else:
        processor, model, prompt_format = load_vlm(VLM_DICT[model_id], return_prompt_format=True)

    # collect aggregate va plots per prompt
    va_aggregate = {}  # key: prompt_slug -> dict with runs / centers / labels
    # Same aggregation for the corresponding split (non-ambiguous control) images.
    # split images have no meaningful boundary, so center is fixed at 0.
    split_aggregate = {}  # key: prompt_slug -> dict with runs / centers / labels
    # Track which va_i produced a valid boundary for each prompt during this run,
    # so we can gate the corresponding split_i tasks (which come later in the
    # iteration order because split_* paths are appended after va_*).
    valid_va_by_prompt: Dict[str, set] = defaultdict(set)
    for image_path, text_prompt in tqdm(tasks, desc='All tasks'):
        image_name = os.path.basename(image_path).split('.')[0]
        prompt, prompt_prefix = text_prompt[0], text_prompt[1]
        prompt_slug = slugify(text_prompt[0]+text_prompt[1])

        # Skip split_i if the corresponding va_i has no valid boundary for this prompt.
        if image_name.startswith("split_") and not chunk_mode:
            suffix = image_name.split("_")[-1]
            if f"va_{suffix}" not in valid_va_by_prompt[prompt_slug]:
                continue

        os.makedirs(os.path.join(results_dir, "text", model_id, image_name), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots", model_id, image_name), exist_ok=True)
        text_path = os.path.join(results_dir, "text", model_id, image_name, f"{prompt_slug}.json")
        angles_img = angles
        if args.center_on_boundary and image_name.startswith("va_"):
            center = get_boundary(boundaries, model_id, image_name, base_prompt_slug)
            if center is None:
                print(f"[SKIP] {model_id} | {image_name}: no default-query boundary to centre on")
                continue
            angles_img = [int(center) + a for a in angles]
        dataset = RotatedRGBAImageDataset(image_path, angles_img)
        if (not args.overwrite) and os.path.exists(text_path):
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

        prob_data=extract_topk_from_file(text_path, k=3, extra_tokens=color_dict.keys())

        if image_name.startswith("va_"):
            x_token, y_token = get_duck_rabbit_tokens(model_id)
            angle_lim = (-45, 45)
            boundary = decision_boundary_from_topk(prob_data, x_token, y_token, angle_lim=angle_lim, plot = True, 
            save_path=os.path.join(results_dir, "plots", model_id, image_name, f"{prompt_slug}_psychometric.png"),
            title=f"{model_id} Q:{prompt} A:{prompt_prefix} {x_token} vs {y_token}",
            )

            if args.plot_only or chunk_mode:
                pass  # plot-only / chunk runs never write boundaries.json (the
                      # fit is recomputed only for centering)
            else:
                update_boundary_json(
                    model_name=model_id,
                    image_name=image_name,
                    prompt=prompt,
                    prompt_prefix=prompt_prefix,
                    prompt_slug=prompt_slug,
                    boundary=boundary,
                    angle_lim=angle_lim,
                    token_a=x_token,
                    token_b=y_token,
                    save_path=os.path.join(bistable_image_dir, "boundaries.json"),
                )

            if boundary is not None:
                center_angle = boundary
                valid_va_by_prompt[prompt_slug].add(image_name)

                # ---- aggregate collection (ONLY if boundary exists) ----
                if prompt_slug not in va_aggregate:
                    va_aggregate[prompt_slug] = {
                        "runs": [],
                        "centers": [],
                        "prompt": prompt,
                        "prefix": prompt_prefix,
                    }
                va_aggregate[prompt_slug]["runs"].append(prob_data)
                va_aggregate[prompt_slug]["centers"].append(center_angle)
            else:
                center_angle = 0
        else:
            center_angle = 0

        # Aggregate split_i prob_data (control images, no center offset).
        if image_name.startswith("split_"):
            if prompt_slug not in split_aggregate:
                split_aggregate[prompt_slug] = {
                    "runs": [],
                    "centers": [],
                    "prompt": prompt,
                    "prefix": prompt_prefix,
                }
            split_aggregate[prompt_slug]["runs"].append(prob_data)
            split_aggregate[prompt_slug]["centers"].append(0)

        # Index by value so that a restricted --angles window (e.g. -45..45) works.
        ambi_idx = angles_img.index(int(center_angle)) if int(center_angle) in angles_img else len(angles_img) // 2
        ambi_image = dataset[ambi_idx]
        plot_topk_tokens(prob_data, 
            save_path=os.path.join(results_dir, "plots", model_id, image_name, f"{prompt_slug}.png"), 
            color_dict=color_dict, xlim=(-45, 45), center_angle=center_angle, max_curves=8)
    del model, processor
    torch.cuda.empty_cache()

    # plot all visual anagrams together (exclude ones without boundary)
    os.makedirs(os.path.join(results_dir, "plots", model_id, "va_all"), exist_ok=True)
    for prompt_slug, pack in va_aggregate.items():
        runs = pack["runs"]
        centers = pack["centers"]

        if len(runs) == 0:
            continue  # no valid va_i with boundary

        plot_topk_tokens(
            runs,  # list of prob_data
            center_angle=centers,  # list of centers
            image=None,
            save_path=os.path.join(results_dir, "plots", model_id, "va_all", f"{prompt_slug}.png"),
            color_dict=color_dict,
            xlim=(-45, 45),
            max_curves=8,
        )

    # plot all split (control) images together
    os.makedirs(os.path.join(results_dir, "plots", model_id, "split_all"), exist_ok=True)
    for prompt_slug, pack in split_aggregate.items():
        runs = pack["runs"]
        centers = pack["centers"]
        if len(runs) == 0:
            continue
        plot_topk_tokens(
            runs,
            center_angle=centers,  # all zeros for split images
            image=None,
            save_path=os.path.join(results_dir, "plots", model_id, "split_all", f"{prompt_slug}.png"),
            color_dict=color_dict,
            xlim=(-45, 45),
            max_curves=8,
        )