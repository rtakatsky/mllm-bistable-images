"""Generate non-illusory split controls (duck + rabbit side by side), style-matched
to a corresponding VA stimulus, with the OpenAI image-edit API.

Each split control is generated per reference VA: the VA's duck view (+45 deg,
composited on white) is passed as the input image to images.edit together with a
"same style" instruction, so split_<name> matches the drawing style of va_<name>.

Downstream: run remove_va_background.py with --rotate 0 on the raw
outputs (split controls are NOT pre-rotated, unlike VA stimuli), then derive
individual duck/rabbit images via alpha-channel connected components (as in
make_masks.py) — only after the splits are confirmed correct.

Reference selection (one of):
  --references path1,path2      explicit reference image paths
  --manifest <judge manifest> --threshold 4 --reference-dir <rgba dir>
                                all candidates that PASS the judge rule

Needs OPENAI_API_KEY. Examples:
    python generate_split_controls.py --references ../data/duck_rabbit/va_0.png
    python generate_split_controls.py \
        --manifest ../outputs/va_judge/manifest_candidates__gpt-5-nano-2025-08-07.json \
        --threshold 4 --reference-dir ../outputs/va_candidates/rgba
"""

import argparse
import base64
import io
import json
import math
import os
import time

from PIL import Image

from utils import OUTPUTS_DIR

EXP_NAME = "split_controls"
PROMPT = ("Using the exact same drawing style, stroke quality, coloring, and level of "
          "detail as this reference image, draw two separate animal heads side by side "
          "on a plain white background: a duck head on the left and a rabbit head on "
          "the right. Two distinct heads, clearly separated, not merged, not hybrid, "
          "not overlapping.")
DUCK_VIEW_ANGLE = 45  # reference VAs are stored -45deg-centered; +45 = duck view


def reference_png(path: str, angle: float) -> bytes:
    """Render a reference view on white and return PNG bytes.

    Same geometry as utils.RotatedRGBAImageDataset (diagonal square canvas
    margin, rotate expand=False, white composite) so the reference is framed
    exactly like the stimuli in the main experiments — only the final 336x336
    downscale is skipped (kept at <=1024 so the style stays legible to the
    image model).
    """
    base = Image.open(path).convert("RGBA")
    w, h = base.size
    diag = math.ceil(math.sqrt(w ** 2 + h ** 2))
    canvas = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    canvas.paste(base, ((diag - w) // 2, (diag - h) // 2), mask=base)
    rotated = canvas.rotate(angle, resample=Image.BICUBIC, expand=False)
    bg = Image.new("RGBA", (diag, diag), (255, 255, 255, 255))
    img = Image.alpha_composite(bg, rotated).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def collect_references(args):
    if args.references:
        return [(os.path.splitext(os.path.basename(p))[0], p)
                for p in args.references.split(",")]
    if args.manifest:
        m = json.load(open(args.manifest))
        k, thr = m["k"], args.threshold
        refs = []
        for name, e in sorted(m["images"].items()):
            v = e["views"]
            if ("45" in v and "-45" in v
                    and v["45"]["counts"]["duck"] >= thr
                    and v["-45"]["counts"]["rabbit"] >= thr):
                refs.append((name, os.path.join(args.reference_dir, f"{name}.png")))
        print(f"[INFO] {len(refs)} references pass the >= {thr}/{k} rule")
        return refs
    raise SystemExit("[ERROR] provide --references or --manifest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=str, default=None,
                        help="comma-separated reference image paths")
    parser.add_argument("--manifest", type=str, default=None,
                        help="judge manifest json; passes become references")
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--reference-dir", type=str, default=None)
    parser.add_argument("--ref-angle", type=float, default=DUCK_VIEW_ANGLE,
                        help="rotation applied to the reference before sending "
                             "(45 for -45-centered VAs; 0 for unrotated references)")
    parser.add_argument("--image-model", type=str, default="gpt-image-1-mini")
    parser.add_argument("--quality", type=str, default="low", choices=["low", "medium", "high"])
    parser.add_argument("--size", type=str, default="1024x1024")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[ERROR] OPENAI_API_KEY is not set")
    from openai import OpenAI
    client = OpenAI()

    raw_dir = os.path.join(OUTPUTS_DIR, EXP_NAME, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    for name, ref_path in collect_references(args):
        out_path = os.path.join(raw_dir, f"split_{name}.png")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[INFO] split_{name}: exists, skipping")
            continue
        if not os.path.exists(ref_path):
            print(f"[WARN] reference missing: {ref_path}, skipping")
            continue
        ref = reference_png(ref_path, args.ref_angle)
        for attempt in range(8):
            try:
                resp = client.images.edit(
                    model=args.image_model,
                    image=(f"{name}.png", ref, "image/png"),
                    prompt=PROMPT,
                    size=args.size,
                    quality=args.quality,
                    n=1,
                )
                break
            except Exception as e:
                if attempt == 7:
                    raise
                wait = min(75, 5 * 2 ** attempt)
                print(f"[WARN] API error ({type(e).__name__}); retry in {wait}s")
                time.sleep(wait)
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(resp.data[0].b64_json))
        with open(out_path.replace(".png", ".json"), "w") as f:
            json.dump({"reference": ref_path, "ref_angle": args.ref_angle,
                       "model": args.image_model, "quality": args.quality,
                       "size": args.size, "prompt": PROMPT}, f, indent=1)
        print(f"[INFO] split_{name}: saved")
        if args.sleep > 0:
            time.sleep(args.sleep)

    print("[INFO] done")
