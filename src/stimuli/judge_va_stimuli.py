"""GPT-5 judge for VA stimulus validity (prefilter), with calibration modes.

Because an API judge may know the duck-rabbit illusion, each request shows ONE
view only, in a fresh context, with a neutral forced-one-word question and no
mention of illusions, ambiguity, rotation, or the other view. Raw answers are
stored verbatim so decision thresholds can be tuned without re-querying;
"both"-type answers (e.g. "duck-rabbit", "ambiguous") are tracked as their own
class — if these show up on single views, the judge is illusion-aware and the
protocol needs revisiting before trusting it on candidates.

Modes (run in this order; do NOT judge candidates until controls/originals
look sane):
  controls    the hand-made duck_0..9 + rabbit_0..9 controls at 0 deg (ground
              truth known -> judge accuracy / format compliance)
  originals   the hand-made va_0..9 stimuli at +45 (duck view) and -45 (rabbit
              view) (these hand-made sets are not part of this repository)
  candidates  --input-dir (default: outputs/va_candidates/rgba)
              at +45 / -45

Needs OPENAI_API_KEY in the environment. Manifest is saved incrementally to
outputs/va_judge/manifest_<mode>.json; already-judged images are
skipped unless --overwrite.

Examples:
    python judge_va_stimuli.py --mode controls
    python judge_va_stimuli.py --mode originals
    python judge_va_stimuli.py --mode candidates
"""

import argparse
import base64
import glob
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from utils import DATA_DIR, OUTPUTS_DIR

EXP_NAME = "va_judge"
CANDIDATE_DIR_DEFAULT = os.path.join(OUTPUTS_DIR, "va_candidates", "rgba")
QUESTION = "What animal is in this drawing? Answer with a single word."
# Views are rendered EXACTLY like utils.RotatedRGBAImageDataset (diagonal square
# canvas margin, rotate expand=False, white composite, 336x336), so the judge
# sees what the models see in the main experiments.
RENDER_TAG = "expmatch336"
OUTPUT_SIZE = (336, 336)

DUCK_WORDS = {"duck", "duckling", "mallard", "goose", "bird", "waterfowl", "swan", "pelican"}
RABBIT_WORDS = {"rabbit", "bunny", "hare"}
BOTH_WORDS = {"duck-rabbit", "rabbit-duck", "both", "ambiguous", "illusion"}


def classify(answer: str) -> str:
    norm = re.sub(r"[^a-z\-]", " ", answer.strip().lower()).strip()
    first = norm.split()[0] if norm.split() else ""
    if first in BOTH_WORDS or norm in BOTH_WORDS:
        return "both"
    if first in DUCK_WORDS:
        return "duck"
    if first in RABBIT_WORDS:
        return "rabbit"
    return "other"


def encode_view(path: str, angle: float) -> str:
    """Mirror utils.RotatedRGBAImageDataset.__getitem__ exactly."""
    base = Image.open(path).convert("RGBA")
    w, h = base.size
    diag = math.ceil(math.sqrt(w ** 2 + h ** 2))
    canvas = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    canvas.paste(base, ((diag - w) // 2, (diag - h) // 2), mask=base)
    rotated = canvas.rotate(angle, resample=Image.BICUBIC, expand=False)
    bg = Image.new("RGBA", (diag, diag), (255, 255, 255, 255))
    composite = Image.alpha_composite(bg, rotated).convert("RGB")
    composite = composite.resize(OUTPUT_SIZE, resample=Image.BICUBIC)
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask_once(client, model: str, b64: str, detail: str) -> str:
    for attempt in range(8):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}",
                                       "detail": detail}},
                        {"type": "text", "text": QUESTION},
                    ],
                }],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # rate limit / transient
            if attempt == 7:
                raise
            wait = min(75, 5 * 2 ** attempt)
            print(f"[WARN] API error ({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)


def judge_image(client, model: str, path: str, angles, k: int, workers: int, detail: str,
                sleep_s: float = 0.0):
    views = {}
    for angle in angles:
        b64 = encode_view(path, angle)
        if sleep_s > 0:  # serial + throttled (low-RPM orgs)
            answers = []
            for _ in range(k):
                answers.append(ask_once(client, model, b64, detail))
                time.sleep(sleep_s)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                answers = list(ex.map(lambda _: ask_once(client, model, b64, detail), range(k)))
        counts = {"duck": 0, "rabbit": 0, "both": 0, "other": 0}
        for a in answers:
            counts[classify(a)] += 1
        views[str(angle)] = {"answers": answers, "counts": counts}
    return views


def collect_images(mode: str, input_dir: str):
    """Return list of (name, path, expected_class_or_None, angles)."""
    dr = os.path.join(DATA_DIR, "duck_rabbit")
    items = []
    if mode == "controls":
        for i in range(10):
            items.append((f"duck_{i}", os.path.join(dr, f"duck_{i}.png"), "duck", [0]))
            items.append((f"rabbit_{i}", os.path.join(dr, f"rabbit_{i}.png"), "rabbit", [0]))
    elif mode == "originals":
        for i in range(10):
            items.append((f"va_{i}", os.path.join(dr, f"va_{i}.png"), None, [45, -45]))
    elif mode == "candidates":
        for path in sorted(glob.glob(os.path.join(input_dir, "*.png"))):
            items.append((os.path.splitext(os.path.basename(path))[0], path, None, [45, -45]))
    elif mode == "singles":
        # QC for derived duck_*/rabbit_* singles: expected class from filename, 0 deg
        for path in sorted(glob.glob(os.path.join(input_dir, "*.png"))):
            base = os.path.splitext(os.path.basename(path))[0]
            expected = base.split("_")[0]
            if expected in ("duck", "rabbit"):
                items.append((base, path, expected, [0]))
    return [it for it in items if os.path.exists(it[1])]


def print_summary(manifest: dict):
    print(f"\n===== summary ({manifest['mode']}, model={manifest['model']}, k={manifest['k']}) =====")
    k = manifest["k"]
    n_pass = {thr: 0 for thr in (0.6, 0.8, 1.0)}
    for name, entry in sorted(manifest["images"].items()):
        parts = []
        for angle, v in entry["views"].items():
            # recompute from raw answers so synonym-set updates apply retroactively
            c = {"duck": 0, "rabbit": 0, "both": 0, "other": 0}
            for a in v["answers"]:
                c[classify(a)] += 1
            v["counts"] = c
            top = max(c, key=c.get)
            parts.append(f"{angle:>4}deg: {top} {c[top]}/{k}" +
                         (f" (both={c['both']})" if c["both"] else ""))
        line = f"{name:12s} " + " | ".join(parts)
        if entry.get("expected"):
            got = max(entry["views"]["0"]["counts"], key=entry["views"]["0"]["counts"].get)
            line += f"  expected={entry['expected']} {'OK' if got == entry['expected'] else 'MISS'}"
        if set(entry["views"].keys()) == {"45", "-45"}:
            for thr in n_pass:
                need = int(k * thr + 1e-9)
                if (entry["views"]["45"]["counts"]["duck"] >= need
                        and entry["views"]["-45"]["counts"]["rabbit"] >= need):
                    n_pass[thr] += 1
        print(line)
    if set(next(iter(manifest["images"].values()))["views"].keys()) == {"45", "-45"}:
        total = len(manifest["images"])
        for thr, n in n_pass.items():
            print(f"pass rate @ >= {thr:.0%} agreement per view: {n}/{total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["controls", "originals", "candidates", "singles"])
    parser.add_argument("--input-dir", type=str, default=CANDIDATE_DIR_DEFAULT)
    parser.add_argument("--judge-model", type=str, default="gpt-5-nano-2025-08-07")
    parser.add_argument("--detail", type=str, default="low", choices=["low", "high", "auto"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds between requests (serializes calls; for low-RPM orgs)")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[ERROR] OPENAI_API_KEY is not set")
    from openai import OpenAI
    client = OpenAI()

    out_dir = os.path.join(OUTPUTS_DIR, EXP_NAME)
    os.makedirs(out_dir, exist_ok=True)
    # manifest is per judge model AND per rendering: never mix protocols
    manifest_path = os.path.join(
        out_dir, f"manifest_{args.mode}__{args.judge_model}__{RENDER_TAG}.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        manifest = json.load(open(manifest_path))
    else:
        manifest = {"mode": args.mode, "model": args.judge_model, "k": args.k,
                    "detail": args.detail, "rendering": RENDER_TAG,
                    "question": QUESTION, "images": {}}

    items = collect_images(args.mode, args.input_dir)
    todo = [it for it in items if it[0] not in manifest["images"]]
    print(f"[INFO] {len(items)} images, {len(todo)} to judge")

    for name, path, expected, angles in todo:
        print(f"[INFO] judging {name}")
        views = judge_image(client, args.judge_model, path, angles, args.k, args.workers,
                            args.detail, args.sleep)
        manifest["images"][name] = {"path": os.path.relpath(path, os.path.dirname(DATA_DIR)),
                                    "expected": expected, "views": views}
        with open(manifest_path, "w") as f:  # incremental, crash-safe
            json.dump(manifest, f, indent=1)

    if manifest["images"]:
        print_summary(manifest)
    print(f"[INFO] manifest: {manifest_path}")
