# Aggregate the per-VA top-down resampling ablation caches
# (x-cue -> d-cue, produced by resampling_ablation --images all-valid --text_only) into a
# va_all MEAN heatmap per model x component, using the same plotting routine as
# the per-pair figures (so the aggregate drops into the existing Fig 10 slots).
# CPU-only: loads processors (not model weights) for tokenization/labels.
import glob
import json
import os

import numpy as np
from transformers import AutoProcessor

from utils import *  # show_ablation_of_text_tokens, centered_va_image, VLM_DICT, DATA_DIR, base_dir, slugify

PROMPT_FORMATS = {
    "llava-1.5-7b": "USER: <image>\n{prompt} ASSISTANT:",
    "llava-1.5-13b": "USER: <image>\n{prompt} ASSISTANT:",
    "llava-v1.6-vicuna-7b": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:",
    "llava-v1.6-mistral-7b": "[INST] <image>\n{prompt} [/INST]",
    "llama3-llava-next-8b": "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
}

X_PROMPT = "List every animal in the image, each in one word. It starts with 'x'."
D_PROMPT = "List every animal in the image, each in one word. It starts with 'd'."
PREFIX = "I see a"
COMPONENTS = ["resid_post", "attn_out", "mlp_out"]

if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--prefix", type=str, default=PREFIX,
                     help="Assistant prefix used by the resampling run whose caches to aggregate "
                          "(selects the cache-directory slugs; output directories inherit them).")
    PREFIX = _ap.parse_args().prefix
    results_dir = os.path.join(OUTPUTS_DIR, "resampling_ablation")
    cache_root = os.path.join(results_dir, "cache")
    bistable_image_dir = os.path.join(DATA_DIR, "duck_rabbit")
    with open(os.path.join(bistable_image_dir, "boundaries.json"), "r", encoding="utf-8") as f:
        boundaries = json.load(f)
    BOUNDARY_SLUG = slugify("List every animal in the image, each in one word." + "I see a")

    x_slug, d_slug = slugify(X_PROMPT + PREFIX), slugify(D_PROMPT + PREFIX)

    for model_id, hf_id in [(m, VLM_DICT[m]) for m in PROMPT_FORMATS]:
        pair_dirs = sorted(glob.glob(os.path.join(cache_root, f"va_s*_va_s*_{x_slug}_{d_slug}_{model_id}")))
        if not pair_dirs:
            print(f"[SKIP] {model_id}: no pair caches found")
            continue

        processor = AutoProcessor.from_pretrained(hf_id)
        tokenizer = processor.tokenizer

        # tokens_of_interest: replicated verbatim from resampling_ablation __main__
        duck_tok = {"llama3-llava-next-8b": "Ġduck"}.get(model_id, "▁du")
        rabbit_tok = {"llava-v1.6-mistral-7b": "▁rab",
                      "llama3-llava-next-8b": "Ġrabbit"}.get(model_id, "▁rabb")
        tokens_of_interest = {}
        for token in [duck_tok, rabbit_tok, "▁bird", "▁D", "▁Rab", "▁in", ".", "▁and"]:
            tid = tokenizer.convert_tokens_to_ids(token)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if tid is None or (unk_id is not None and tid == unk_id):
                alt = token.replace("▁", "Ġ") if "▁" in token else token.replace("Ġ", "▁")
                alt_id = tokenizer.convert_tokens_to_ids(alt)
                if alt_id is not None and alt_id != unk_id:
                    token, tid = alt, alt_id
                else:
                    continue
            tokens_of_interest[token] = tid

        target_prompt = PROMPT_FORMATS[model_id].format(prompt=D_PROMPT) + PREFIX

        # real post-image text-token count/labels from processor-expanded ids,
        # using one valid centered VA image (any valid one; geometry identical)
        va_name = os.path.basename(pair_dirs[0])[: len("va_s000")]
        img = centered_va_image(os.path.join(bistable_image_dir, f"{va_name}.png"),
                                boundaries, model_id, va_name, BOUNDARY_SLUG)
        ids = processor(text=target_prompt, images=img, return_tensors="pt")["input_ids"][0].tolist()
        # image token id: the id whose count is largest (placeholder run) — robust
        # across llava variants without loading the model config
        from collections import Counter
        img_tok_id, img_tok_count = Counter(ids).most_common(1)[0]
        last_img_pos = max(i for i, t in enumerate(ids) if t == img_tok_id)
        n_post = len(ids) - (last_img_pos + 1)
        labels = [tokenizer.decode([t]) for t in ids[-n_post:]]
        print(f"{model_id}: {len(pair_dirs)} pairs, S={len(ids)}, img_tokens={img_tok_count}, n_post={n_post}")

        for component in COMPONENTS:
            stores, origs, used = [], [], 0
            for d in pair_dirs:
                p = os.path.join(d, f"{component}.npy")
                pt = os.path.join(d, f"{component}__textonly.npy")
                po = os.path.join(d, "original.npy")
                src = p if os.path.exists(p) else pt
                if not (os.path.exists(src) and os.path.exists(po)):
                    print(f"[WARN] missing {component} in {os.path.basename(d)}")
                    continue
                s, o = np.load(src), np.load(po)
                if stores and s.shape != stores[0].shape:
                    print(f"[WARN] shape mismatch in {os.path.basename(d)}: {s.shape} vs {stores[0].shape}")
                    continue
                if s.shape[-1] != len(tokens_of_interest):
                    print(f"[WARN] token-dim mismatch in {os.path.basename(d)}: {s.shape[-1]} vs {len(tokens_of_interest)}")
                    continue
                stores.append(s)
                origs.append(o)
                used += 1
            if not stores:
                print(f"[SKIP] {model_id}/{component}: nothing to aggregate")
                continue
            mean_store = np.nanmean(np.stack(stores), axis=0)
            mean_orig = np.mean(np.stack(origs), axis=0)

            save_dir = os.path.join(results_dir, "plots", component, "text_tokens",
                                    model_id, "va_all_va_all", f"{x_slug}_{d_slug}")
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "mean_store.npy"), mean_store)
            np.save(os.path.join(save_dir, "mean_original.npy"), mean_orig)
            show_ablation_of_text_tokens(
                tokenizer,
                target_prompt,
                mean_store,
                mean_orig,
                tokens_of_interest,
                show_diff=[[0, 1], [-3, -1]],
                cache_key=component,
                save_dir=save_dir,
                text_prompt_length=n_post,
                text_token_labels=labels,
            )
            print(f"  {component}: aggregated {used} pairs -> {save_dir}")
