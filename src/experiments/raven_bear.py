# Figure-ground analyses on the raven--bear image.
# Three analyses, all on the unrotated image (no boundary needed):
#   1) Exclusivity under beam search (with single-token bird/other prefixes).
#   2) Red-circle bottom-up modulation.
#   3) Top-down single-letter prefix cues.

import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import argparse
import logging

import numpy as np
import torch
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.ERROR)


IMAGE_NAME = "raven_bear"


# Single-letter top-down cues. "b" hits bird/bear, "d" hits dog, "x" is a
# control letter that doesn't match any expected output. (No "r" cue: the
# models report bird vs dog/bear and never "raven".)
TOP_DOWN_PREFIXES = ["b", "d", "x"]


def _make_color_dict():
    # Tokens that may appear in beam continuations or top-k logits.
    return {
        "▁bird":   "skyblue",
        "Ġbird":   "skyblue",
        "▁dog":    "orange",
        "Ġdog":    "orange",
        "▁bear":   "salmon",
        "Ġbear":   "salmon",
        "▁raven":  "darkgreen",
        "Ġraven":  "darkgreen",
        "▁and":    "gray",
        "Ġand":    "gray",
    }


if __name__ == "__main__":
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading and inference; only regenerate plots from cached results (CPU-friendly).")
    parser.add_argument("--base_dir", type=str,
                        default=PROJECT_ROOT)
    args = parser.parse_args()
    model_id = args.model

    image_path  = os.path.join(args.base_dir, "data/duck_rabbit", f"{IMAGE_NAME}.png")
    results_dir = os.path.join(args.base_dir, "outputs", IMAGE_NAME)

    tok_a, tok_b = get_raven_bear_tokens(model_id)
    color_dict   = _make_color_dict()

    seed = 42
    print(f"Running {model_id} on {IMAGE_NAME}...")
    if args.plot_only:
        processor = model = prompt_format = None
        print("[PLOT-ONLY] skipping model load; replotting cached results")
    else:
        processor, model, prompt_format = load_vlm(VLM_DICT[model_id], return_prompt_format=True)
        # load_vlm leaves the model on CPU; with fully cached beam results
        # run_vlm_sampling (which moves it to GPU as a side effect) is never called,
        # so teacher-forced scoring would silently run on CPU.
        model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval()

    # =====================================================================
    # 1) Beam search: continuations with each token forced in the prefix,
    #    swept over rotation (±45°) as a bottom-up manipulation. Follows the
    #    rotation_beam main-experiment pipeline: canonical beam strings are taken
    #    from the 0° beam search and teacher-force-scored at every angle so
    #    their probability curves are complete.
    # =====================================================================
    import importlib
    beam_mod = importlib.import_module("experiments.rotation_beam")

    base_prompt = "List every animal in the image, each in one word."
    # (prefix_text, label_for_filename). Prefix is appended to the chat prompt.
    beam_specs = [
        ("I see a",              "i-see-a"),
        (f"I see a {tok_a.lstrip('▁Ġ')}",  f"i-see-a-{tok_a.lstrip('▁Ġ')}"),
        (f"I see a {tok_b.lstrip('▁Ġ')}",  f"i-see-a-{tok_b.lstrip('▁Ġ')}"),
    ]

    beam_angles = list(range(-45, 46))

    for prefix_text, label in beam_specs:
        prompt_slug = f"list-every-animal-in-the-image-each-in-one-word-{label}"
        text_path = os.path.join(results_dir, "text",  model_id, "beam_search", f"{prompt_slug}.json")
        center_text_path = os.path.join(results_dir, "text", model_id, "beam_search", f"{prompt_slug}__center.json")
        plot_path = os.path.join(results_dir, "plots", model_id, "beam_search", f"{prompt_slug}.png")
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # 1a) Center-only beam search (canonical top beam strings at 0°).
        center_dataset = RotatedRGBAImageDataset(image_path, [0])
        if os.path.exists(center_text_path) and (args.plot_only or not args.overwrite):
            with open(center_text_path, "r", encoding="utf-8") as f:
                center_results = json.load(f)
        elif args.plot_only:
            print(f"[PLOT-ONLY SKIP] missing cache: {center_text_path}")
            continue
        else:
            center_results = run_vlm_sampling(
                model, processor, center_dataset, base_prompt,
                sample_n=1, temperature=1.0, max_new_tokens=4,
                seed=seed, batch_size=1,
                use_logit=False, top_k=10,
                prompt_format=prompt_format,
                prompt_prefix=prefix_text,
                num_beams=4,
            )
            with open(center_text_path, "w", encoding="utf-8") as f:
                json.dump(center_results, f, ensure_ascii=False, indent=2)
        # (string, ids, prob) triples; ids are needed because decoded strings lose
        # their leading space, which made string-based re-scoring collapse
        # word-initial continuations by orders of magnitude.
        center_cands = beam_mod.extract_center_candidates(
            center_results, 0, None if args.plot_only else processor.tokenizer.eos_token_id
        )
        center_strings = [s for s, _, _ in center_cands]

        # 1b) All-angle beam search.
        dataset = RotatedRGBAImageDataset(image_path, beam_angles)
        stripped = False
        if os.path.exists(text_path) and (args.plot_only or not args.overwrite):
            with open(text_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            if not args.plot_only:
                # drop entries appended by string-based scoring
                # (never under plot-only: the file is not re-appended/saved then)
                stripped = beam_mod.strip_appended_entries(results, 4)
            print(f"[SKIP] {text_path}")
        elif args.plot_only:
            print(f"[PLOT-ONLY SKIP] missing cache: {text_path}")
            continue
        else:
            results = run_vlm_sampling(
                model, processor, dataset, base_prompt,
                sample_n=1, temperature=1.0, max_new_tokens=4,
                seed=seed, batch_size=1,
                use_logit=False, top_k=10,
                prompt_format=prompt_format,
                prompt_prefix=prefix_text,
                num_beams=4,
            )

        # 1c) Score the 0° beam strings at every angle (fills curve gaps).
        modified = False if args.plot_only else beam_mod.append_candidates_to_beam_results(
            results=results,
            dataset=dataset,
            model=model,
            processor=processor,
            prompt=base_prompt,
            prompt_format=prompt_format,
            prompt_prefix=prefix_text,
            candidates=[(s, ids) for s, ids, _ in center_cands],
        )
        if (not os.path.exists(text_path)) or modified or stripped:
            with open(text_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        # 1d) Plot (rotation_beam style, top-k anchored at 0°).
        prob_data = extract_topk_from_file(
            text_path,
            k=5,
            extra_tokens=center_strings,
            anchor_angle=0,
        )
        plot_topk_tokens(
            prob_data,
            save_path=plot_path,
            image=None,
            color_dict=color_dict,
            xlim=(-45, 45),
            center_angle=0,
        )

    # =====================================================================
    # 2) Red-circle bottom-up modulation.
    # =====================================================================
    red_prompt        = "List every animal in the image, each in one word."
    red_prompt_prefix = "I see a"
    red_slug = f"list-every-animal-in-the-image-each-in-one-word-i-see-a_{tok_a}_{tok_b}_r0p071_s0p143_lw4"
    red_text_path = os.path.join(results_dir, "text",  model_id, "red_circle", f"{red_slug}.json")
    red_plot_path = os.path.join(results_dir, "plots", model_id, "red_circle", f"{red_slug}.png")
    os.makedirs(os.path.dirname(red_text_path), exist_ok=True)
    os.makedirs(os.path.dirname(red_plot_path), exist_ok=True)

    # Frame the image exactly as RotatedRGBAImageDataset presents it at angle 0
    # (centered on the sqrt(2)-diagonal canvas, composited on white), so the
    # red-circle stage sees the same framing as the beam/top-down stages and
    # the circle grid spans the same canvas as in red_circle runs.
    framed = RotatedRGBAImageDataset(image_path, [0], output_size=None)[0]
    framed_path = os.path.join(results_dir, "cache", f"{IMAGE_NAME}_framed.png")
    os.makedirs(os.path.dirname(framed_path), exist_ok=True)
    framed.save(framed_path)

    # Results restored from the archive may carry '#Uxxxx'-mangled token
    # characters in their names; fall back to that spelling when present.
    _mangled = red_text_path.replace("\u2581", "#U2581").replace("\u0120", "#U0120")
    if (not os.path.exists(red_text_path)) and os.path.exists(_mangled):
        red_text_path = _mangled
    if os.path.exists(red_text_path) and (args.plot_only or not args.overwrite):
        print(f"[SKIP] {red_text_path}")
    elif args.plot_only:
        print(f"[PLOT-ONLY SKIP] missing cache: {red_text_path}")
        red_text_path = None
    else:
        red_dataset = RedCircleImageDataset(
            framed,
            radius_ratio=1/14,
            stride_ratio=1/7,
            line_width=4,
        )
        red_results = run_vlm_sampling_red_circle(
            model, processor, red_dataset, red_prompt,
            sample_n=1, temperature=1.0, max_new_tokens=1,
            seed=seed, batch_size=2,
            use_logit=True, top_k=100,
            prompt_format=prompt_format,
            prompt_prefix=red_prompt_prefix,
            extra_tokens=[tok_a, tok_b],
        )
        with open(red_text_path, "w") as f:
            json.dump(red_results, f, ensure_ascii=False, indent=2)

    if red_text_path is not None:
      plot_heatmaps(
        red_text_path,
        framed_path,
        group1_tokens=[tok_b],
        group2_tokens=[tok_a],
        save_path=red_plot_path,
        group1_label=tok_b.lstrip("▁Ġ").capitalize(),
        group2_label=tok_a.lstrip("▁Ġ").capitalize(),
      )

    # =====================================================================
    # 3) Top-down single-letter prefix cues.
    #    Logit-diff (tok_a vs tok_b) under each cue prompt.
    # =====================================================================
    # "No cue" baseline for the cue plot: the red-circle run's baseline
    # sample (center=None) uses the same prompt, prefix, and framed image,
    # just without any cue sentence.
    cue_to_record = {}
    with open(red_text_path) as f:  # noqa: red_text_path resolved above (mangled fallback)
        for item in json.load(f):
            if item.get("center", "missing") is None and item.get("output"):
                base = item["output"][0]
                cue_to_record["no cue"] = {
                    "prob_data": [{
                        "angle": 0,
                        "tokens": base.get("tokens", []),
                        "probs": base.get("probs", []),
                    }],
                    "center_angle": 0,
                }
                break

    # Beam continuations under each cue at the centre angle. On figure--ground
    # images exclusivity is weak, so the first token alone does not represent the
    # report (the model may go on to name the other animal); these runs let the
    # cue effect be read with the same single/multiple classification used for
    # the duck--rabbit aggregates.
    for cue in [None] + TOP_DOWN_PREFIXES:   # None = no-cue baseline, same settings
        cue_prompt = ("List every animal in the image, each in one word." if cue is None
                      else f"List every animal in the image, each in one word. It starts with '{cue}'.")
        slug = slugify(cue_prompt + "I see a")
        beam_path = os.path.join(results_dir, "text", model_id, "top_down_beam", f"{slug}.json")
        os.makedirs(os.path.dirname(beam_path), exist_ok=True)
        if os.path.exists(beam_path) and (args.plot_only or not args.overwrite):
            print(f"[SKIP] {beam_path}")
        elif args.plot_only:
            print(f"[PLOT-ONLY SKIP] missing cache: {beam_path}")
        else:
            beam_res = run_vlm_sampling(
                model, processor, RotatedRGBAImageDataset(image_path, [0]), cue_prompt,
                sample_n=1, temperature=1.0, max_new_tokens=12,
                seed=seed, batch_size=1,
                use_logit=False, top_k=8,
                prompt_format=prompt_format,
                prompt_prefix="I see a",
                num_beams=8,
            )
            with open(beam_path, "w", encoding="utf-8") as f:
                json.dump(beam_res, f, ensure_ascii=False, indent=2)
            print(f"[OK] {beam_path}")

    td_results = {}
    for cue in TOP_DOWN_PREFIXES:
        cue_prompt = f"List every animal in the image, each in one word. It starts with '{cue}'."
        cue_prompt_prefix = "I see a"
        slug = slugify(cue_prompt + cue_prompt_prefix)
        text_path = os.path.join(results_dir, "text",  model_id, "top_down", f"{slug}.json")
        os.makedirs(os.path.dirname(text_path), exist_ok=True)

        if os.path.exists(text_path) and (args.plot_only or not args.overwrite):
            print(f"[SKIP] {text_path}")
        elif args.plot_only:
            print(f"[PLOT-ONLY SKIP] missing cache: {text_path}")
            continue
        else:
            dataset = RotatedRGBAImageDataset(image_path, [0])
            results = run_vlm_sampling(
                model, processor, dataset, cue_prompt,
                sample_n=1, temperature=1.0, max_new_tokens=1,
                seed=seed, batch_size=1,
                use_logit=True, top_k=100,
                prompt_format=prompt_format,
                prompt_prefix=cue_prompt_prefix,
                extra_tokens=[tok_a, tok_b],
            )
            with open(text_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        prob_data = extract_topk_from_file(text_path, k=5, extra_tokens=[tok_a, tok_b])
        cue_to_record[cue] = {"prob_data": prob_data, "center_angle": 0}
        # Collect tok_a / tok_b probabilities at angle 0 for the probs JSON below.
        for record in prob_data:
            if record.get("angle") != 0:
                continue
            probs = dict(zip(record.get("tokens", []), record.get("probs", [])))
            td_results[cue] = {
                tok_a: float(probs.get(tok_a, 0.0)),
                tok_b: float(probs.get(tok_b, 0.0)),
            }
            break

    # Combined cue plot in the main-experiment style (top_down_cues).
    td_plot_path = os.path.join(
        results_dir, "plots", model_id, "top_down",
        "list-every-animal-in-the-image-each-in-one-word-i-see-a_prefix.png",
    )
    plot_cue_effects_single(
        cue_to_record,
        cue_type="prefix",
        cue_order=TOP_DOWN_PREFIXES,
        color_dict=color_dict,
        title=f"{model_id} | {IMAGE_NAME} | prefix",
        save_path=td_plot_path,
    )

    # Persist the raw numbers next to the plot.
    td_probs_path = os.path.join(
        results_dir, "probs", model_id, "top_down",
        "list-every-animal-in-the-image-each-in-one-word-i-see-a_prefix.json",
    )
    os.makedirs(os.path.dirname(td_probs_path), exist_ok=True)
    with open(td_probs_path, "w") as f:
        json.dump(td_results, f, ensure_ascii=False, indent=2)

    del model, processor
    torch.cuda.empty_cache()
