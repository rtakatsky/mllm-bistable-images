# Object-counting query for bistable images.
# Asks "How many objects are in the image? Answer with a number only." and records
# (1) the first-token digit distribution after a " " prefix and (2) a no-prefix
# beam-search answer distribution, for harper, raven_bear, every Visual Anagram
# (at its model-specific boundary angle) and every split_s* control (angle 0).
# Per-model group means land in tables/<model>/summary.json (appendix table).
import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import argparse
import logging
import glob
import re

import numpy as np
import torch
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.ERROR)


if __name__ == "__main__":
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading and inference; only regenerate plots from cached results (CPU-friendly).")
    parser.add_argument("--images", type=str, default="harper,raven_bear,va,split",
                        help="Comma-separated stimulus names or groups ('va' = every va_s* with a boundary, "
                             "evaluated at its boundary angle; 'split' = every split_s* control at angle 0).")
    args = parser.parse_args()
    model_id = args.model

    bistable_image_dir = os.path.join(DATA_DIR, 'duck_rabbit')
    results_dir = os.path.join(OUTPUTS_DIR, 'object_counting')
    os.makedirs(results_dir, exist_ok=True)

    # Stimulus groups: the canonical duck-rabbit, the
    # raven-bear figure-ground image, every Visual Anagram with a boundary for
    # this model (evaluated AT its boundary angle, like every other VA aggregate)
    # and every style-matched split_s* control (angle 0).
    groups = {}
    for g in args.images.split(","):
        if g == "va":
            groups["va"] = sorted(os.path.basename(x)[:-4] for x in glob.glob(os.path.join(bistable_image_dir, "va_s*.png")))
        elif g == "split":
            groups["split"] = sorted(os.path.basename(x)[:-4] for x in glob.glob(os.path.join(bistable_image_dir, "split_s*.png")))
        else:
            groups[g] = [g]

    with open(os.path.join(bistable_image_dir, "boundaries.json"), "r") as f:
        boundaries = json.load(f)
    boundary_prompt_slug = slugify("List every animal in the image, each in one word." + "I see a")

    count_prompt = "How many objects are in the image? Answer with a number only."
    # (1) first-token measure: the " " prefix ends the assistant turn with an
    # explicit whitespace token (SP "▁" / BPE "Ġ") so the first generated token
    # is the digit itself (" 1" tokenizes as [▁/Ġ, 1] for all five LLaVA
    # tokenizers (digit mass >= 0.997 after the prefix).
    first_token_prefix = " "
    # (2) beam measure (validation, no prefix): free beam search over the natural
    # answer; each beam is mapped to the first integer (or number word) it
    # contains and the EOS-excluded beam probabilities are turned into
    # prefix-disjoint mass shares (same convention as classify_beam_longcont).
    beam_num_beams = 8
    beam_max_new_tokens = 4

    prompt_slug = slugify(count_prompt + first_token_prefix)
    beam_slug = prompt_slug + "__beam"

    color_dict = {"1": "blue", "2": "orange", "3": "red"}
    seed = 42
    batch_size = 2
    top_k = 100

    NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                    "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    def beam_answer(text):
        """First integer (or number word) in a beam string; None if absent."""
        t = text.strip().lower()
        for tok in re.findall(r"\d+|[a-z]+", t):
            if tok.isdigit():
                return int(tok)
            if tok in NUMBER_WORDS:
                return NUMBER_WORDS[tok]
        return None

    def beam_shares(tokens, probs):
        """Prefix-disjoint, normalized mass per integer answer ('none' = no number)."""
        w = np.array(probs, dtype=float)
        texts = [t.strip() for t in tokens]
        w_adj = w.copy()
        for i_a, ta in enumerate(texts):
            for i_b, tb in enumerate(texts):
                if i_a != i_b and len(tb) > len(ta) and tb.startswith(ta) and (len(ta) == 0 or not tb[len(ta)].isalnum()):
                    w_adj[i_a] -= w[i_b]
        w = np.clip(w_adj, 0.0, None)
        cov = float(w.sum())
        shares = {}
        if cov > 0:
            for t, wi in zip(tokens, w / cov):
                key = beam_answer(t)
                key = "none" if key is None else str(key)
                shares[key] = shares.get(key, 0.0) + float(wi)
        return shares, cov

    print(f"Running {model_id}...")
    if args.plot_only:
        processor = model = prompt_format = None
        print("[PLOT-ONLY] skipping model load; replotting cached results")
    else:
        processor, model, prompt_format = load_vlm(VLM_DICT[model_id], return_prompt_format=True)

    per_image = {}
    for group, names in groups.items():
        for image_name in tqdm(names, desc=f"{group}"):
            image_path = os.path.join(bistable_image_dir, f"{image_name}.png")
            if not os.path.exists(image_path):
                print(f"[SKIP] missing image {image_path}")
                continue
            if image_name.startswith("va_s"):
                angle = get_boundary(boundaries, model_id, image_name, boundary_prompt_slug)
                if angle is None:
                    print(f"[SKIP] {model_id} | {image_name}: boundary not found, skipping")
                    continue
                angle = int(round(angle))
            else:
                angle = 0

            text_path = os.path.join(results_dir, "text",  model_id, image_name, f"{prompt_slug}.json")
            beam_path = os.path.join(results_dir, "text",  model_id, image_name, f"{beam_slug}.json")
            prob_path = os.path.join(results_dir, "probs", model_id, image_name, f"{prompt_slug}.json")
            plot_path = os.path.join(results_dir, "plots", model_id, image_name, f"{prompt_slug}.png")
            for p in (text_path, prob_path, plot_path):
                os.makedirs(os.path.dirname(p), exist_ok=True)

            # ---- (1) first-token logits with the " " prefix ----
            if os.path.exists(text_path) and (args.plot_only or not args.overwrite):
                print(f"[SKIP] {model_id} results already exist: {text_path}")
            elif args.plot_only:
                print(f"[PLOT-ONLY SKIP] missing cache: {text_path}")
                continue
            else:
                dataset = RotatedRGBAImageDataset(image_path, [angle])
                results = run_vlm_sampling(
                    model, processor, dataset, count_prompt,
                    sample_n=1, temperature=1.0, max_new_tokens=1,
                    seed=seed, batch_size=batch_size,
                    use_logit=True, top_k=top_k,
                    prompt_format=prompt_format,
                    prompt_prefix=first_token_prefix,
                    extra_tokens=list(color_dict.keys()),
                )
                with open(text_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            # ---- (2) beam search without prefix ----
            if os.path.exists(beam_path) and (args.plot_only or not args.overwrite):
                pass
            elif args.plot_only:
                print(f"[PLOT-ONLY SKIP] missing beam cache: {beam_path}")
            else:
                dataset = RotatedRGBAImageDataset(image_path, [angle])
                beam_results = run_vlm_sampling(
                    model, processor, dataset, count_prompt,
                    sample_n=1, temperature=1.0, max_new_tokens=beam_max_new_tokens,
                    seed=seed, batch_size=1,
                    use_logit=False, top_k=beam_num_beams,
                    prompt_format=prompt_format,
                    prompt_prefix=None,
                    num_beams=beam_num_beams,
                )
                with open(beam_path, "w") as f:
                    json.dump(beam_results, f, ensure_ascii=False, indent=2)

            # ---- per-image summary ----
            prob_data = extract_topk_from_file(text_path, k=5, extra_tokens=list(color_dict.keys()))
            if group in ("harper", "raven_bear"):
                plot_topk_tokens(prob_data, save_path=plot_path, color_dict=color_dict, xlim=(-1, 1))
            with open(text_path) as f:
                raw = json.load(f)[0]["output"][0]
            digit_probs = {}
            for tok, pr in zip(raw["tokens"], raw["probs"]):
                core = tok.strip("▁Ġ ")
                if core.isdigit():
                    digit_probs[core] = digit_probs.get(core, 0.0) + float(pr)
            first = {k: digit_probs.get(k, 0.0) for k in ("1", "2", "3")}
            first["digit_mass"] = float(sum(digit_probs.values()))
            first["top100_mass"] = float(sum(raw["probs"]))

            beam = None
            if os.path.exists(beam_path):
                with open(beam_path) as f:
                    b = json.load(f)[0]["output"][0]
                shares, cov = beam_shares(b["tokens"], b["probs"])
                beam = {"shares": shares, "coverage": cov,
                        "beams": [(t, float(pr)) for t, pr in zip(b["tokens"], b["probs"])]}

            with open(prob_path, "w") as f:
                json.dump({"angle": angle, "probs": {k: first[k] for k in ("1", "2", "3")},
                           "first_token": first, "beam": beam}, f, ensure_ascii=False, indent=2)
            per_image[image_name] = {"group": group, "angle": angle, "first_token": first, "beam": beam}

    # ---- per-model table: group means (+-SD) of the two measures ----
    table_dir = os.path.join(results_dir, "tables", model_id)
    os.makedirs(table_dir, exist_ok=True)
    summary = {"model": model_id, "prompt": count_prompt, "first_token_prefix": first_token_prefix,
               "beam": {"num_beams": beam_num_beams, "max_new_tokens": beam_max_new_tokens},
               "groups": {}, "images": per_image}
    for group in groups:
        rows = [v for v in per_image.values() if v["group"] == group]
        if not rows:
            continue
        g = {"n": len(rows)}
        for key in ("1", "2", "3", "digit_mass"):
            vals = [r["first_token"][key] for r in rows]
            g[f"first_p{key}_mean"] = float(np.mean(vals)); g[f"first_p{key}_std"] = float(np.std(vals))
        for key in ("1", "2", "none"):
            vals = [r["beam"]["shares"].get(key, 0.0) for r in rows if r["beam"] is not None]
            if vals:
                g[f"beam_{key}_mean"] = float(np.mean(vals)); g[f"beam_{key}_std"] = float(np.std(vals))
        covs = [r["beam"]["coverage"] for r in rows if r["beam"] is not None]
        if covs:
            g["beam_coverage_mean"] = float(np.mean(covs))
        summary["groups"][group] = g
        print(f"{model_id:24s} {group:10s} n={g['n']:3d} first p1={g['first_p1_mean']:.3f}+-{g['first_p1_std']:.3f} "
              f"p2={g['first_p2_mean']:.3f} | beam p1={g.get('beam_1_mean', float('nan')):.3f} p2={g.get('beam_2_mean', float('nan')):.3f} "
              f"none={g.get('beam_none_mean', float('nan')):.3f} cov={g.get('beam_coverage_mean', float('nan')):.2f}")
    with open(os.path.join(table_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if model is not None:
        del model, processor
        torch.cuda.empty_cache()
