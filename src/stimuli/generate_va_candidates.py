"""Generate duck-rabbit Visual Anagram candidates with DeepFloyd IF.

Faithful port of the generation loop of the original Colab notebook
(cells 9/11/20): IF-I-M + IF-II-M + SD-x4-upscaler, views ['identity',
'rotate_cw'], prompts "drawing of a duck head" / "drawing of a rabbit head",
30 steps, guidance 10.0, noise_level 50, per-seed torch.Generator. The saved
candidate is the 1024px *identity* view; background removal and the -45deg
centering rotation happen later in remove_va_background.py.

Requires a GPU (see slurm/README.md) and a
HuggingFace token with the DeepFloyd license accepted (huggingface-cli login,
or HF_TOKEN in the environment).

Example (one chunk of seeds):
    python generate_va_candidates.py --seed-start 0 --seed-end 10
"""

import argparse
import json
import os

# utils sets HF_HOME (repo-local cache unless VLM_BISTABLE_HF_CACHE_DIR / HF_HOME
# is set) before any huggingface/transformers/diffusers import.
from utils import HF_CACHE_DIR, OUTPUTS_DIR

import torch
from PIL import Image
from diffusers import DiffusionPipeline

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2

EXP_NAME = "va_candidates"
PROMPT_1 = "drawing of a duck head"      # identity view
PROMPT_2 = "drawing of a rabbit head"    # rotate_cw view
VIEW_NAMES = ["identity", "rotate_cw"]
STAGE_1_ID = "DeepFloyd/IF-I-M-v1.0"
STAGE_2_ID = "DeepFloyd/IF-II-M-v1.0"
STAGE_3_ID = "stabilityai/stable-diffusion-x4-upscaler"


def im_to_pil(im: torch.Tensor) -> Image.Image:
    """(C,H,W) in [-1,1] -> PIL. Same normalization as the notebook's im_to_np."""
    im = (im / 2 + 0.5).clamp(0, 1)
    im = im.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((im * 255).round().astype("uint8"))


def load_pipelines(device: str, cpu_offload: bool):
    stage_1 = DiffusionPipeline.from_pretrained(
        STAGE_1_ID, variant="fp16", torch_dtype=torch.float16, cache_dir=HF_CACHE_DIR
    )
    stage_2 = DiffusionPipeline.from_pretrained(
        STAGE_2_ID, text_encoder=None, variant="fp16", torch_dtype=torch.float16,
        cache_dir=HF_CACHE_DIR,
    )
    stage_3 = DiffusionPipeline.from_pretrained(
        STAGE_3_ID, torch_dtype=torch.float16, cache_dir=HF_CACHE_DIR
    )
    for stage in (stage_1, stage_2, stage_3):
        if cpu_offload:
            stage.enable_model_cpu_offload()
        else:
            stage.to(device)
    return stage_1, stage_2, stage_3


def generate_candidate(stage_1, stage_2, stage_3, views, seed: int, device: str):
    gen = torch.Generator(device=device).manual_seed(seed)

    prompts = [PROMPT_1, PROMPT_2]
    prompt_embeds = [stage_1.encode_prompt(p) for p in prompts]
    prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
    prompt_embeds = torch.cat(prompt_embeds)
    negative_prompt_embeds = torch.cat(negative_prompt_embeds)

    image_64 = sample_stage_1(
        stage_1, prompt_embeds, negative_prompt_embeds, views,
        num_inference_steps=30, guidance_scale=10.0, reduction="mean", generator=gen,
    )
    image_256 = sample_stage_2(
        stage_2, image_64, prompt_embeds, negative_prompt_embeds, views,
        num_inference_steps=30, guidance_scale=10.0, reduction="mean",
        noise_level=50, generator=gen,
    )
    image_1024 = stage_3(
        prompt=prompts[0], image=image_256, noise_level=0,
        output_type="pt", generator=gen,
    ).images
    image_1024 = image_1024 * 2 - 1
    return image_1024[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=10, help="exclusive")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    device = "cuda"
    out_dir = os.path.join(OUTPUTS_DIR, EXP_NAME)
    raw_dir = os.path.join(out_dir, "raw")
    preview_dir = os.path.join(out_dir, "plots")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    todo = [
        seed for seed in range(args.seed_start, args.seed_end)
        if args.overwrite or not os.path.exists(os.path.join(raw_dir, f"seed_{seed:03d}.png"))
    ]
    print(f"[INFO] seeds {args.seed_start}..{args.seed_end - 1}: {len(todo)} to generate")
    if not todo:
        raise SystemExit(0)

    stage_1, stage_2, stage_3 = load_pipelines(device, args.cpu_offload)
    views = get_views(VIEW_NAMES)

    for seed in todo:
        print(f"[INFO] generating seed {seed}")
        image = generate_candidate(stage_1, stage_2, stage_3, views, seed, device)

        identity = im_to_pil(views[0].view(image))
        identity.save(os.path.join(raw_dir, f"seed_{seed:03d}.png"))

        # side-by-side preview of both views for quick browsing
        rotcw = im_to_pil(views[1].view(image))
        preview = Image.new("RGB", (identity.width + rotcw.width, identity.height), "white")
        preview.paste(identity, (0, 0))
        preview.paste(rotcw, (identity.width, 0))
        preview.save(os.path.join(preview_dir, f"seed_{seed:03d}_views.png"))

        meta = {
            "seed": seed,
            "prompt_1": PROMPT_1,
            "prompt_2": PROMPT_2,
            "views": VIEW_NAMES,
            "stage_1": STAGE_1_ID,
            "stage_2": STAGE_2_ID,
            "stage_3": STAGE_3_ID,
            "num_inference_steps": 30,
            "guidance_scale": 10.0,
            "stage_2_noise_level": 50,
        }
        with open(os.path.join(raw_dir, f"seed_{seed:03d}.json"), "w") as f:
            json.dump(meta, f, indent=1)

    print("[INFO] done")
