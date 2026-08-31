"""Background removal + centering rotation for VA candidates.

For each raw candidate (identity view, duck-side up):
  rembg (u2net) -> RGBA with transparent background -> rotate --rotate degrees
  (default -45) -> save.

Also writes a contact sheet (plots/) showing raw | duck view (+45) | rabbit
view (-45) composited on white, for quick manual QC.

CPU-only; runs fine on a login node (no SLURM needed). The u2net weights
(~170 MB) download on first use into the repo cache (U2NET_HOME below), NOT
the shared home directory.

Example:
    python remove_va_background.py
    python remove_va_background.py --input-dir <dir> --output-dir <dir>  # e.g. split controls
"""

import argparse
import glob
import os

from utils import OUTPUTS_DIR, HF_CACHE_DIR

# Keep u2net weights inside the model cache (same root as the HF cache).
os.environ.setdefault("U2NET_HOME", os.path.join(HF_CACHE_DIR, "u2net"))

from PIL import Image
from rembg import new_session, remove

EXP_NAME = "va_candidates"
QC_VIEW_ANGLES = (45, -45)  # +45 = duck (identity) view, -45 = rabbit (rotate_cw) view
QC_TILE = 256


def on_white(img: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")


def qc_row(rgba: Image.Image, raw: Image.Image) -> list:
    tiles = [raw.convert("RGB")]
    for angle in QC_VIEW_ANGLES:
        tiles.append(on_white(rgba.rotate(angle, expand=True)))
    out = []
    for t in tiles:
        t = t.copy()
        t.thumbnail((QC_TILE, QC_TILE))
        tile = Image.new("RGB", (QC_TILE, QC_TILE), "white")
        tile.paste(t, ((QC_TILE - t.width) // 2, (QC_TILE - t.height) // 2))
        out.append(tile)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str,
                        default=os.path.join(OUTPUTS_DIR, EXP_NAME, "raw"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(OUTPUTS_DIR, EXP_NAME, "rgba"))
    parser.add_argument("--rotate", type=float, default=-45.0,
                        help="post-removal rotation (deg, PIL convention)")
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True,
                        help="crop to alpha bounding box after rotation")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(os.path.dirname(args.output_dir.rstrip("/")), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not paths:
        raise SystemExit(f"[WARN] no PNGs in {args.input_dir}")
    print(f"[INFO] {len(paths)} images from {args.input_dir}")

    session = new_session("u2net")
    rows = []
    for path in paths:
        name = os.path.basename(path)
        out_path = os.path.join(args.output_dir, name)
        raw = Image.open(path).convert("RGB")
        if os.path.exists(out_path) and not args.overwrite:
            rgba = Image.open(out_path).convert("RGBA")
        else:
            rgba = remove(raw, session=session).convert("RGBA")
            rgba = rgba.rotate(args.rotate, expand=True)
            if args.crop:
                # crop to alpha bbox so the object isn't tiny in the canvas
                # (RotatedRGBAImageDataset re-adds margins when rotating)
                bbox = rgba.getchannel("A").getbbox()
                if bbox:
                    rgba = rgba.crop(bbox)
            rgba.save(out_path)
            print(f"[INFO] {name}: saved {rgba.size}")
        rows.append((name, qc_row(rgba, raw)))

    # contact sheet: one row per image (raw | +45 on white | -45 on white)
    n_cols = 1 + len(QC_VIEW_ANGLES)
    label_h = 18
    sheet = Image.new("RGB", (n_cols * QC_TILE, len(rows) * (QC_TILE + label_h)), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for r, (name, tiles) in enumerate(rows):
        y = r * (QC_TILE + label_h)
        draw.text((4, y + 2), f"{name}   [raw | +45 duck-view | -45 rabbit-view]", fill="black")
        for c, tile in enumerate(tiles):
            sheet.paste(tile, (c * QC_TILE, y + label_h))
    sheet_path = os.path.join(plots_dir, "bg_removal_contact_sheet.png")
    sheet.save(sheet_path)
    print(f"[INFO] contact sheet: {sheet_path}")
