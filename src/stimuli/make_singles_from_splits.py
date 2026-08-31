"""Derive single-object duck/rabbit images from split controls.

For each split (RGBA, duck head left / rabbit head right, two disconnected
alpha components): produce duck_<name>.png and rabbit_<name>.png by zeroing the
alpha of the OTHER object's component. The canvas is unchanged — no cropping —
so object position and scale match the split exactly (position-matched
controls), as in the original duck_i/rabbit_i (same size as split_i).

Small alpha specks (<1% of opaque area) are removed from both outputs.

Example:
    python make_singles_from_splits.py
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy import ndimage

from utils import OUTPUTS_DIR

SPLIT_DIR_DEFAULT = os.path.join(OUTPUTS_DIR, "split_controls", "rgba")
SINGLES_DIR_DEFAULT = os.path.join(OUTPUTS_DIR, "split_controls", "singles")


def derive_singles(split_path: str):
    img = Image.open(split_path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3] > 128
    labels, n = ndimage.label(alpha)
    if n == 0:
        raise ValueError("empty alpha channel")
    sizes = ndimage.sum(alpha, labels, range(1, n + 1))
    major = [i + 1 for i, s in enumerate(sizes) if s > 0.01 * alpha.sum()]
    if len(major) != 2:
        raise ValueError(f"expected 2 major components, found {len(major)}")
    # left component = duck, right = rabbit (per generation prompt)
    cx = {lab: ndimage.center_of_mass(labels == lab)[1] for lab in major}
    duck_lab, rabbit_lab = sorted(major, key=lambda lab: cx[lab])

    out = {}
    for animal, keep in (("duck", duck_lab), ("rabbit", rabbit_lab)):
        a = arr.copy()
        a[..., 3] = np.where(labels == keep, arr[..., 3], 0)
        out[animal] = Image.fromarray(a)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=SPLIT_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=SINGLES_DIR_DEFAULT)
    parser.add_argument("--pattern", type=str, default="split_seed_*.png")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print(f"[INFO] {len(paths)} splits from {args.input_dir}")
    n_ok = 0
    for path in paths:
        name = os.path.basename(path).replace("split_", "").replace(".png", "")
        outs = {a: os.path.join(args.output_dir, f"{a}_{name}.png") for a in ("duck", "rabbit")}
        if all(os.path.exists(p) for p in outs.values()) and not args.overwrite:
            n_ok += 1
            continue
        try:
            singles = derive_singles(path)
        except ValueError as e:
            print(f"[WARN] {name}: {e}, skipping")
            continue
        for animal, img in singles.items():
            img.save(outs[animal])
        n_ok += 1
        print(f"[INFO] {name}: duck + rabbit saved")
    print(f"[INFO] done ({n_ok}/{len(paths)} ok)")
