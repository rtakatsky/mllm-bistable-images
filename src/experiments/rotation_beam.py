import argparse
import glob
import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.ERROR)


def build_text_prompt_for_scoring(
    processor,
    prompt: str,
    prompt_format: Optional[str] = None,
    prompt_prefix: Optional[str] = None,
) -> str:
    """
    Rebuild the exact text prompt used in run_vlm_sampling, so sequence scoring matches generation.
    """
    if prompt_format is None:
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image'}
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    else:
        text_prompt = prompt_format.format(prompt=prompt)
        if prompt_prefix is not None:
            text_prompt = text_prompt + prompt_prefix
    return text_prompt


def trim_candidate_ids(ids: List[int], eos_id: Optional[int]) -> List[int]:
    """
    Keep ids up to the first EOS (inclusive). generate() pads finished beams with
    eos/pad ids whose transition scores are masked to zero, so teacher-forcing the
    padded tail would double-count p(eos).
    """
    out = []
    for t in ids:
        out.append(t)
        if eos_id is not None and t == eos_id:
            break
    return out


def extract_center_candidates(
    results: List[Dict[str, Any]],
    center_angle: Union[int, float],
    eos_id: Optional[int],
) -> List[Tuple[str, List[int], float]]:
    """
    (string, token_ids, prob) triples from the beam entry at angle == center_angle.
    Token ids are kept alongside the decoded string because decoding is lossy:
    batch_decode strips the leading space of word-initial continuations.
    """
    entry = None
    for item in results:
        if item.get("angle") == center_angle and item.get("output"):
            entry = item["output"][0]
            break
    if entry is None:
        return []

    cands = []
    seen = set()
    for s, ids, p in zip(entry.get("tokens", []), entry.get("indices", []), entry.get("probs", [])):
        if not isinstance(s, str) or s.strip() == "" or not ids:
            continue
        if s in seen:
            continue
        seen.add(s)
        cands.append((s, trim_candidate_ids(list(ids), eos_id), float(p)))
    return cands


def strip_appended_entries(results: List[Dict[str, Any]], n_native: int) -> bool:
    """
    Drop teacher-forced entries (appended after the native beams) from
    cached results. Word-initial strings were scored on re-tokenized ids that lose
    their leading space ('▁and ...' -> 'and ...'), collapsing their probabilities
    by orders of magnitude, so all appended values must be recomputed; the first
    n_native entries are raw beam-search outputs and are unaffected.
    """
    changed = False
    for item in results:
        for out in item.get("output", []):
            if len(out.get("tokens", [])) > n_native:
                for key in ("tokens", "indices", "logits", "probs"):
                    if isinstance(out.get(key), list):
                        out[key] = out[key][:n_native]
                changed = True
    return changed


@torch.no_grad()
def score_candidate_ids_for_image(
    model,
    processor,
    image,
    base_text_prompt: str,
    candidates: List[Tuple[str, List[int]]],
    chunk_size: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """
    Teacher-forced joint probability of exact beam token ids for a single image.

    Scoring decoded strings instead is wrong for word-initial continuations:
    base + "and a rabbit" re-tokenizes 'and' without the leading-space piece
    ('▁and'), so the model is scored on a token sequence it never generated.
    """
    device = next(model.parameters()).device
    if len(candidates) == 0:
        return {}

    prompt_inputs = processor(
        images=[image],
        text=[base_text_prompt],
        return_tensors="pt",
    )
    prompt_ids = prompt_inputs["input_ids"][0]
    prompt_len = prompt_ids.shape[0]
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    scored = {}
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start:start + chunk_size]
        B = len(chunk)
        max_cont = max(len(ids) for _, ids in chunk)
        T = prompt_len + max_cont

        input_ids = torch.full((B, T), pad_id, dtype=prompt_ids.dtype)
        attn = torch.zeros((B, T), dtype=torch.long)
        for i, (_, ids) in enumerate(chunk):
            L = prompt_len + len(ids)
            input_ids[i, :prompt_len] = prompt_ids
            input_ids[i, prompt_len:L] = torch.tensor(ids, dtype=prompt_ids.dtype)
            attn[i, :L] = 1

        batch = {}
        for k, v in prompt_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            # Replicate image tensors along the batch dim. dim0-concat matches each
            # processor's batching convention (incl. Qwen2-VL's flattened patches).
            batch[k] = torch.cat([v] * B, dim=0).to(device) if torch.is_tensor(v) else v
        batch["input_ids"] = input_ids.to(device)
        batch["attention_mask"] = attn.to(device)

        logits = model(**batch).logits  # (B, T, V)
        # Only continuation positions matter; slice before softmax to keep memory small.
        step_logits = logits[:, prompt_len - 1:T - 1, :].float()
        log_probs = torch.log_softmax(step_logits, dim=-1)

        for i, (s, ids) in enumerate(chunk):
            n = len(ids)
            ids_t = torch.tensor(ids, device=device)
            token_lp = log_probs[i, :n, :].gather(-1, ids_t.unsqueeze(-1)).squeeze(-1)
            seq_logprob = float(token_lp.sum().detach().cpu())
            scored[s] = {
                "indices": list(ids),
                "logprob": seq_logprob,
                "prob": float(np.exp(seq_logprob)),
            }
        del logits, step_logits, log_probs
    return scored


def append_candidates_to_beam_results(
    results: List[Dict[str, Any]],
    dataset,
    model,
    processor,
    prompt: str,
    prompt_format: Optional[str],
    prompt_prefix: Optional[str],
    candidates: List[Tuple[str, List[int]]],
) -> bool:
    """
    For every angle, teacher-force score candidates missing from the tracked list and
    append them, so downstream extraction sees a complete (no-NaN) matrix.
    Returns whether anything was appended.
    """
    if not candidates:
        return False

    base_text_prompt = build_text_prompt_for_scoring(
        processor=processor,
        prompt=prompt,
        prompt_format=prompt_format,
        prompt_prefix=prompt_prefix,
    )

    modified = False
    for idx, item in enumerate(results):
        out_list = item.get("output", [])
        if not out_list:
            continue
        beam_entry = out_list[0]
        toks = beam_entry.get("tokens", [])
        if not isinstance(toks, list):
            continue

        existing = set(toks)
        missing = [(s, ids) for s, ids in candidates if s not in existing]
        if len(missing) == 0:
            continue

        image = dataset[idx]  # RotatedRGBAImageDataset returns PIL image
        scored = score_candidate_ids_for_image(
            model=model,
            processor=processor,
            image=image,
            base_text_prompt=base_text_prompt,
            candidates=missing,
        )

        for key in ("indices", "logits", "probs"):
            if not isinstance(beam_entry.get(key), list):
                beam_entry[key] = []

        for s, _ in missing:
            if s not in scored:
                continue
            beam_entry["tokens"].append(s)
            beam_entry["indices"].append(scored[s]["indices"])   # list[int]
            beam_entry["logits"].append(scored[s]["logprob"])    # log prob
            beam_entry["probs"].append(scored[s]["prob"])        # prob
            modified = True

    return modified


def slice_prob_data(run: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Keep only the first k tracked continuations (they are stored in ranked order)."""
    return [
        {"angle": it["angle"], "tokens": it["tokens"][:k], "probs": it["probs"][:k]}
        for it in run
    ]


if __name__ == "__main__":
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading, inference, stripping, and candidate appending; only regenerate plots from cached results (CPU-friendly).")
    # Pilot / long-continuation aggregate design:
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--angles", type=str, default=None,
                        help="Comma-separated RELATIVE angles (e.g. '-30,0,30'); default: -45..45 at 1 deg")
    parser.add_argument("--prompts", type=str, default="all", choices=["all", "two-animals-pair", "topdown-dr"],
                        help="'two-animals-pair' = base list prompt + 'There are two animals.' variant only; "
                             "'topdown-dr' = base list prompt + the 'd'/'r' prefix cues of the top_down_cues top-down experiment")
    parser.add_argument("--results-subdir", type=str, default=None,
                        help="Write under outputs/<subdir> instead of the default experiment dir")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Only run beam search + per-image JSONs; skip candidate pooling/appends/plots")
    parser.add_argument("--images", type=str, default=None,
                        help="Comma-separated stimulus names (e.g. 'harper,split') to restrict the run; "
                             "default: the full stimulus set. Aggregate (va_all/split_all) plots are only "
                             "rebuilt for classes present in the restricted set.")
    args = parser.parse_args()
    model_id = args.model

    bistable_image_dir = os.path.join(DATA_DIR, 'duck_rabbit')
    bistable_image_paths = [os.path.join(bistable_image_dir, f"{image_name}.png") for image_name in ["harper", "split"]]
    # The Visual Anagram set (see data/duck_rabbit/va_set_manifest.json).
    bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "va_s*.png")))
    # split_s* included for all seeds; the loop filters out ones whose va_s*
    # has no boundary for the current prompt.
    bistable_image_paths += sorted(glob.glob(os.path.join(bistable_image_dir, "split_s*.png")))

    if args.images is not None:
        wanted = set(args.images.split(","))
        # Allow stimuli outside the duck-rabbit set (e.g. raven_bear for the
        # figure-ground appendix) as long as <name>.png exists; non-va names
        # get center angle 0 below.
        listed = {os.path.basename(p).split(".")[0] for p in bistable_image_paths}
        for name in sorted(wanted - listed):
            extra = os.path.join(bistable_image_dir, f"{name}.png")
            if os.path.exists(extra):
                bistable_image_paths.append(extra)
        bistable_image_paths = [p for p in bistable_image_paths
                                if os.path.basename(p).split(".")[0] in wanted]
        missing = wanted - {os.path.basename(p).split(".")[0] for p in bistable_image_paths}
        if missing:
            raise ValueError(f"--images names not found in {bistable_image_dir}: {sorted(missing)}")

    results_dir = os.path.join(OUTPUTS_DIR, 'rotation_beam')
    if args.results_subdir:
        results_dir = os.path.join(OUTPUTS_DIR, args.results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    angle_lim = 45
    rel_angles = np.arange(-angle_lim, angle_lim + 1, 1).tolist()  # relative window (inclusive)
    if args.angles:
        rel_angles = [int(x) for x in args.angles.split(',')]

    prompts = [ # might include a first answer but not used for getting the boundary
        # ["List every animal in the image, each in one word.", "There's a"],
        # ["List every object in the image, each in one word.", "I see", ""],
        ["List every object in the image, each in one word.", "I see", ""],
        ["List every animal in the image, each in one word.", "I see a", ""],
        ["List every animal in the image, each in one word.", "I see a", " bird"],
        ["List every animal in the image, each in one word.", "I see a", " rabbit"],
        ["List every animal in the image, each in one word. There are two animals.", "I see a", ""],
    ]
    if args.prompts == "two-animals-pair":
        prompts = [
            ["List every animal in the image, each in one word.", "I see a", ""],
            ["List every animal in the image, each in one word. There are two animals.", "I see a", ""],
        ]
    elif args.prompts == "topdown-dr":
        # Report-level version of the top-down boundary test:
        # the same cue strings as top_down_cues (td_prompt), scored by beam-continuation
        prompts = [
            ["List every animal in the image, each in one word.", "I see a", ""],
            ["List every animal in the image, each in one word. It starts with 'd'.", "I see a", ""],
            ["List every animal in the image, each in one word. It starts with 'r'.", "I see a", ""],
        ]

    color_dict = {
        "▁du": "blue",
        "▁Duck": "blue",
        "▁duck": "blue",
        "Ġduck": "blue",
        "▁bird": "skyblue",
        "Ġbird": "skyblue",
        "▁rabb": "orange",
        "▁Rab": "orange",
        "▁rab": "orange",
        "Ġrabbit": "orange",
        "▁b": "salmon",
        "▁Bun": "salmon",
        "Ġbunny": "salmon",
        "▁drawing": "gray",
        "Ġdrawing": "gray",
        "▁fish": "green",
        "Ġfish": "green",
    }

    sample_n = 1
    temperature = 1.0
    max_new_tokens = args.max_new_tokens
    seed = 42
    batch_size = 1
    use_logit = False
    top_k = 10
    num_beams = args.num_beams

    # Continuation aggregation across samples: pool each sample's center-angle beams,
    # rank by mean center probability (absent -> 0), keep K_SUFF in the JSONs/tables
    # and draw only the top K_VIS in the aggregate plots.
    K_SUFF = 10
    K_VIS = 5
    # Entries per angle produced by beam search itself; anything beyond this in a
    # cached file was teacher-force appended by an earlier (buggy) run and is stripped.
    n_native = min(top_k, num_beams)

    os.makedirs(os.path.join(results_dir, "text"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)

    # boundaries (model-aware)
    boundaries_path = os.path.join(bistable_image_dir, "boundaries.json")
    if os.path.exists(boundaries_path):
        with open(boundaries_path, "r", encoding="utf-8") as f:
            boundaries = json.load(f)
    else:
        boundaries = {}
        print(f"[WARN] boundaries file not found: {boundaries_path}")

    print(f"Running {model_id}...")
    if args.plot_only:
        processor = model = prompt_format = None
        eos_id = None
        print("[PLOT-ONLY] skipping model load; will only replot cached results")
    else:
        processor, model, prompt_format = load_vlm(VLM_DICT[model_id], return_prompt_format=True)
        # load_vlm leaves the model on CPU; historically run_vlm_sampling moved it to
        # GPU as a side effect, but with fully cached beam results it is never called
        # and teacher-forced scoring would silently run on CPU.
        model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval()
        eos_id = processor.tokenizer.eos_token_id

    for text_prompt in tqdm(prompts, desc='Prompts'):
        prompt, prompt_prefix = text_prompt[0], text_prompt[1] + text_prompt[2]
        # Boundary centering always uses the base task question + the "I see a"
        # prefix under which boundaries.json entries were fitted (rotation_first_token).
        # Slugifying the full prompt would give appended-cue prompts (e.g.
        # "... There are two animals.") a slug with no boundary entry, silently
        # skipping every va_* image for those prompts (same convention as
        # top_down_cues's base_prompt_slug).
        base_question = text_prompt[0]
        for b in ("List every animal in the image, each in one word.",
                  "List every object in the image, each in one word."):
            if base_question.startswith(b):
                base_question = b
                break
        prompt_slug_for_boundary = slugify(base_question + "I see a")
        if len(text_prompt) == 2:
            prompt_slug = prompt_slug_for_boundary
        elif len(text_prompt) == 3:
            prompt_slug = slugify(text_prompt[0] + text_prompt[1] + text_prompt[2])

        # ---- Phase 1: beam-search results per image (cached or fresh) ----
        records = []
        for image_path in tqdm(bistable_image_paths, desc=f'{prompt_slug} (beam)', leave=False):
            image_name = os.path.basename(image_path).split('.')[0]

            out_path = os.path.join(results_dir, "text", model_id, image_name, f"{prompt_slug}.json")
            plot_path = os.path.join(results_dir, "plots", model_id, image_name, f"{prompt_slug}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

            # Skip split_i if the corresponding va_i has no boundary for this prompt.
            if image_name.startswith("split_"):
                suffix = image_name.split("_")[-1]
                if get_boundary(boundaries, model_id, f"va_{suffix}", prompt_slug_for_boundary) is None:
                    print(f"[SKIP] No corresponding va boundary for {image_name} | {prompt_slug_for_boundary}")
                    continue

            # center angle: model-specific boundary for va images, 0 otherwise
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

            angles = [center_angle + da for da in rel_angles]

            stripped = False
            if (not args.overwrite) and os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                if not args.plot_only:
                    # plot-only must not strip: the file is never re-appended/saved
                    stripped = strip_appended_entries(results, n_native)
                print(f"[SKIP] {model_id} results already exist: {out_path}")
            elif args.plot_only:
                print(f"[PLOT-ONLY SKIP] missing cache: {out_path}")
                continue
            else:
                dataset = RotatedRGBAImageDataset(image_path, angles)
                results = run_vlm_sampling(
                    model, processor, dataset, prompt,
                    sample_n=sample_n, temperature=temperature, max_new_tokens=max_new_tokens,
                    seed=seed, batch_size=batch_size,
                    use_logit=use_logit, top_k=top_k,
                    prompt_format=prompt_format,
                    prompt_prefix=prompt_prefix,
                    num_beams=num_beams,
                )
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            records.append({
                "image_name": image_name,
                "image_path": image_path,
                "out_path": out_path,
                "plot_path": plot_path,
                "center_angle": center_angle,
                "angles": angles,
                "results": results,
                "stripped": stripped,
                # own center-angle (relative 0) beams, same anchoring as harper's angle 0
                "center_cands": extract_center_candidates(results, center_angle, eos_id),
            })

        if args.phase1_only:
            continue

        # ---- Phase 2: global top-k continuations per image class ----
        # Pool every sample's center-angle beams, rank by mean center probability
        # (samples where a string is not in the beam list contribute 0 — a lower
        # bound below their k-th beam's probability), keep top K_SUFF.
        global_cands = {}
        for cls in ("va", "split"):
            cls_recs = [r for r in records if r["image_name"].startswith(f"{cls}_s")]
            if len(cls_recs) == 0:
                global_cands[cls] = []
                continue

            pool = {}  # string -> {"ids": [...], "probs": {image_name: p}}
            for r in cls_recs:
                for s, ids, p in r["center_cands"]:
                    if s not in pool:
                        pool[s] = {"ids": ids, "probs": {}}
                    pool[s]["probs"][r["image_name"]] = p

            ranked = sorted(
                pool.items(),
                key=lambda kv: sum(kv[1]["probs"].values()) / len(cls_recs),
                reverse=True,
            )[:K_SUFF]
            global_cands[cls] = [(s, info["ids"]) for s, info in ranked]

            table_path = os.path.join(
                results_dir, "tables", model_id, f"{prompt_slug}__{cls}_global_continuations.json"
            )
            os.makedirs(os.path.dirname(table_path), exist_ok=True)
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump({
                    "prompt_slug": prompt_slug,
                    "class": cls,
                    "n_images": len(cls_recs),
                    "images": [r["image_name"] for r in cls_recs],
                    "continuations": [
                        {
                            "string": s,
                            "ids": info["ids"],
                            "mean_center_prob": sum(info["probs"].values()) / len(cls_recs),
                            "n_present": len(info["probs"]),
                        }
                        for s, info in ranked
                    ],
                }, f, ensure_ascii=False, indent=2)
            print(f"[GLOBAL] {prompt_slug} {cls}: "
                  + ", ".join(repr(s) for s, _ in global_cands[cls][:K_VIS]))

        # ---- Phase 3: fill in missing values, per-image plots, aggregate collection ----
        va_runs, va_centers = [], []
        split_runs, split_centers = [], []
        for r in tqdm(records, desc=f'{prompt_slug} (scoring)', leave=False):
            image_name = r["image_name"]
            if image_name.startswith("va_s"):
                cls = "va"
            elif image_name.startswith("split_s"):
                cls = "split"
            else:
                cls = None

            # own center beams (for the per-image plot) + class-global continuations
            extras = []
            seen = set()
            for s, ids, _ in r["center_cands"]:
                if s not in seen:
                    seen.add(s)
                    extras.append((s, ids))
            if cls is not None:
                for s, ids in global_cands[cls]:
                    if s not in seen:
                        seen.add(s)
                        extras.append((s, ids))

            if args.plot_only:
                modified = False
            else:
                dataset = RotatedRGBAImageDataset(r["image_path"], r["angles"])
                modified = append_candidates_to_beam_results(
                results=r["results"],
                dataset=dataset,
                model=model,
                processor=processor,
                prompt=prompt,
                prompt_format=prompt_format,
                prompt_prefix=prompt_prefix,
                candidates=extras,
            )

            if modified or r["stripped"] or (not os.path.exists(r["out_path"])):
                with open(r["out_path"], 'w', encoding='utf-8') as f:
                    json.dump(r["results"], f, ensure_ascii=False, indent=2)

            # per-image plot: top-k anchored at the center angle, as before
            prob_data = extract_topk_from_file(
                r["out_path"],
                k=min(5, top_k),
                extra_tokens=[s for s, _, _ in r["center_cands"]],
                anchor_angle=r["center_angle"],
            )
            plot_topk_tokens(
                prob_data,
                save_path=r["plot_path"],
                image=None,
                color_dict=color_dict,
                xlim=(-angle_lim, angle_lim),
                center_angle=r["center_angle"],
                beam_mode=True,
                max_curves=8,
            )

            # aggregate collection: exactly the global continuation set, ranked order,
            # complete at every angle after the append above
            if cls is not None and len(global_cands[cls]) > 0:
                agg_data = extract_topk_from_file(
                    r["out_path"],
                    k=0,
                    extra_tokens=[s for s, _ in global_cands[cls]],
                    anchor_angle=r["center_angle"],
                )
                if cls == "va":
                    va_runs.append(agg_data)
                    va_centers.append(r["center_angle"])
                else:
                    split_runs.append(agg_data)
                    split_centers.append(0)

        # ---- Phase 4: aggregate plots (mean +/- std across samples, top K_VIS) ----
        for cls_name, runs, centers in (("va_all", va_runs, va_centers),
                                        ("split_all", split_runs, split_centers)):
            if len(runs) == 0:
                continue
            agg_plot_path = os.path.join(results_dir, "plots", model_id, cls_name, f"{prompt_slug}.png")
            os.makedirs(os.path.dirname(agg_plot_path), exist_ok=True)
            plot_topk_tokens(
                runs,
                center_angle=centers,
                image=None,
                save_path=agg_plot_path,
                color_dict=color_dict,
                xlim=(-angle_lim, angle_lim),
                beam_mode=True,
                class_sum=True,
            )

    del model, processor
    torch.cuda.empty_cache()