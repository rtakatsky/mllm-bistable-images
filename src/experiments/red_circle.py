import glob
import os

from utils import PROJECT_ROOT, HF_CACHE_DIR  # noqa: F401  (utils sets HF_HOME before transformers is imported)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


import math
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from typing import List, Dict, Optional, Union, Literal
import random
import numpy as np
from tqdm import tqdm
import argparse
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

if __name__ == "__main__":
    from utils import *
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--radius_ratio", type=float, default=1/14)
    parser.add_argument("--stride_ratio", type=float, default=1/7)
    parser.add_argument("--line_width", type=int, default=4)
    parser.add_argument("--base_dir", type=str, default=PROJECT_ROOT)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model loading and inference; only regenerate plots from cached results (CPU-friendly).")
    args = parser.parse_args()
    model_id = args.model

    bistable_image_dir = os.path.join(args.base_dir, "data/duck_rabbit")
    bistable_image_paths = [os.path.join(bistable_image_dir, f"{image_name}.png") for image_name in ["harper"]]
    results_dir = os.path.join(args.base_dir, "outputs/red_circle")
    os.makedirs(results_dir, exist_ok=True)

    radius_ratio = args.radius_ratio
    stride_ratio = args.stride_ratio
    line_width = args.line_width

    r_tag = float_tag(radius_ratio, ndigits=3)
    s_tag = float_tag(stride_ratio, ndigits=3)
    lw_tag = f"{line_width}"

    prompts = [
        # ["List every animal in the image, each in one word.", "There's a"],
        ["List every animal in the image, each in one word.", "I see a"],
    ]

    sample_n=1
    temperature=1.0
    max_new_tokens=1
    seed=42
    batch_size=2
    use_logit=True
    top_k=100
    tasks = [
        (image_path, prompt)
        for image_path in bistable_image_paths
        for prompt in prompts
    ]

    print(f"Running {model_id}...")
    if args.plot_only:
        processor = model = prompt_format = None
        print("[PLOT-ONLY] skipping model load; replotting cached results")
    else:
        processor, model, prompt_format = load_vlm(VLM_DICT[model_id], return_prompt_format=True)
    for image_path, prompt in tqdm(tasks, desc='All tasks'):
        image_name = os.path.basename(image_path).split('.')[0]
        prompt_slug = slugify(prompt[0]+prompt[1])
        tok1, tok2 = get_duck_rabbit_tokens(model_id)
        fname_stem = f"{prompt_slug}_{tok1}_{tok2}_r{r_tag}_s{s_tag}_lw{lw_tag}"
        out_path = os.path.join(results_dir, "text", model_id, image_name, f"{fname_stem}.json")

        if args.plot_only:
            if not os.path.exists(out_path):
                print(f"[PLOT-ONLY SKIP] missing cache: {out_path}")
                continue
        elif (not args.overwrite) and os.path.exists(out_path):
            print(f"[SKIP] {model_id} results already exist: {out_path}")
            continue
        else:
            dataset = RedCircleImageDataset(image_path, radius_ratio=radius_ratio, stride_ratio=stride_ratio, line_width=line_width)

            results = run_vlm_sampling_red_circle(
                model, processor, dataset, prompt[0],
                sample_n=sample_n, temperature=temperature, max_new_tokens=max_new_tokens,
                seed=seed, batch_size=batch_size,
                use_logit=use_logit, top_k=top_k,
                prompt_format=prompt_format,
                prompt_prefix=prompt[1],
                extra_tokens=[tok1, tok2],
            )

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        plot_heatmaps(out_path, image_path, group1_tokens=[tok2], group2_tokens=[tok1],\
             save_path=os.path.join(results_dir, "plots", model_id, image_name, f"{fname_stem}.png"), 
            #  title=f"{model_id}: {prompt[0]+prompt[1]}",
             )
    del model, processor
    torch.cuda.empty_cache()
