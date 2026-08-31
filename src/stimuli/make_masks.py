# One-shot data prep: split data/duck_rabbit/split_{i}.png into
# data/duck_rabbit/duck_{i}.png and data/duck_rabbit/rabbit_{i}.png by masking
# out one animal at a time. Used in the top-down-cues experiment to test whether
# the model can override visual evidence with prefix cues alone.
#
# Method: connected components on the alpha channel.
#   1) Treat opaque pixels as foreground.
#   2) Label connected components.
#   3) Sort by centroid x; find the largest gap in centroid_x as the
#      left/right boundary (duck on the left, rabbit on the right).
#   4) For duck_i.png: set alpha=0 on the rabbit components.
#      For rabbit_i.png: set alpha=0 on the duck components.

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def split_components_into_animals(alpha_binary: np.ndarray):
    """Return (duck_mask, rabbit_mask) — boolean (H, W) arrays."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        alpha_binary.astype(np.uint8), connectivity=8
    )

    # Drop background (label 0). Component labels 1..n-1.
    if n <= 2:
        # 0 components (empty image) or 1 component (only background or only one blob).
        return np.zeros_like(alpha_binary, dtype=bool), np.zeros_like(alpha_binary, dtype=bool)

    comp_labels = np.arange(1, n)
    comp_centroids_x = centroids[1:, 0]  # x-coordinate of each component centroid

    # Sort components by centroid_x; find the largest gap.
    order = np.argsort(comp_centroids_x)
    sorted_xs = comp_centroids_x[order]
    if len(sorted_xs) < 2:
        return np.zeros_like(alpha_binary, dtype=bool), np.zeros_like(alpha_binary, dtype=bool)
    gaps = np.diff(sorted_xs)
    split_idx = int(np.argmax(gaps))  # boundary between the leftmost and rightmost clusters

    duck_labels   = comp_labels[order[: split_idx + 1]]
    rabbit_labels = comp_labels[order[split_idx + 1 :]]

    duck_mask   = np.isin(labels, duck_labels)
    rabbit_mask = np.isin(labels, rabbit_labels)
    return duck_mask, rabbit_mask


def make_masks(split_path: str, duck_path: str, rabbit_path: str) -> bool:
    img = Image.open(split_path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    binary = alpha > 0

    duck_mask, rabbit_mask = split_components_into_animals(binary)
    if not duck_mask.any() or not rabbit_mask.any():
        print(f"[WARN] {split_path}: could not separate two animals (got "
              f"{duck_mask.sum()} duck pixels, {rabbit_mask.sum()} rabbit pixels)")
        return False

    duck_arr = arr.copy()
    duck_arr[rabbit_mask, 3] = 0
    Image.fromarray(duck_arr).save(duck_path)

    rabbit_arr = arr.copy()
    rabbit_arr[duck_mask, 3] = 0
    Image.fromarray(rabbit_arr).save(rabbit_path)
    return True


if __name__ == "__main__":
    from utils import DATA_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(DATA_DIR, "duck_rabbit"),
    )
    parser.add_argument("--n", type=int, default=10, help="process split_0 .. split_{n-1}")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    n_made = 0
    n_skip = 0
    for i in range(args.n):
        split_path  = os.path.join(args.data_dir, f"split_{i}.png")
        duck_path   = os.path.join(args.data_dir, f"duck_{i}.png")
        rabbit_path = os.path.join(args.data_dir, f"rabbit_{i}.png")
        if not os.path.exists(split_path):
            print(f"[MISSING] {split_path}")
            continue
        if (not args.overwrite) and os.path.exists(duck_path) and os.path.exists(rabbit_path):
            print(f"[SKIP] duck_{i}.png and rabbit_{i}.png already exist")
            n_skip += 1
            continue
        if make_masks(split_path, duck_path, rabbit_path):
            print(f"[OK] split_{i}.png -> duck_{i}.png, rabbit_{i}.png")
            n_made += 1

    print(f"\nDone: {n_made} new pair(s), {n_skip} skipped.")
