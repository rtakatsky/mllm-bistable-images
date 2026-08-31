# Compose the paper's figures as single multi-panel matplotlib figures. CPU-only: every panel is re-drawn from the plot-input
# caches the experiment scripts already persist (text/probs/tables/cache JSON,
# npz, npy) -- no model is loaded (the resampling heatmaps load a *processor*
# for token labels, as aggregate_resampling.py does).
#
# Why: the per-model PNGs under outputs/*/plots are drawn 8-10 in wide with
# default 10 pt text, then LaTeX shrinks five of them into \textwidth, leaving
# ~2 pt text. Here each paper figure is ONE figure of the final physical width
# (6.5 in for figure*, 3.0 in for a column), so fonts render at their true
# size, and rows of per-model panels share one legend / axis labels where the
# entries are identical across models (classified beam, free-form classes,
# DLA components, scatter series). Legends whose entries are model-specific
# (rotation top-k tokens, beam continuation strings, top-down tokens) stay
# per panel, placed under the panel.
#
# Output: outputs/panels/<figure>.png (the versions used in the paper are
# shipped under figures/panels/). Run:  cd src && python -m figures.make_paper_panels
# (optionally --only <substring> to regenerate a subset).
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

from utils import (  # noqa: E402
    PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR, VLM_DICT,
    extract_topk_from_file, get_boundary, get_duck_rabbit_tokens, get_raven_bear_tokens,
    classify_beam_continuation, build_token_angle_map, slugify, centered_va_image,
    dominant_patch_map_from_logitlens,
    RotatedRGBAImageDataset, _upsample_grid_bilinear, _stable_color, NEUTRAL_PALETTE,
)

OUT_DIR = os.path.join(OUTPUTS_DIR, "panels")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
STIM_DIR = os.path.join(DATA_DIR, "duck_rabbit")

LLAVA5 = ["llava-1.5-7b", "llava-1.5-13b", "llava-v1.6-vicuna-7b", "llava-v1.6-mistral-7b", "llama3-llava-next-8b"]
# Models whose result dirs are absent. With --allow-missing these are dropped
# from every row and listed at the end; by default a missing model is a hard
# error, so a paper figure is never silently regenerated with a model missing.
ALLOW_MISSING = False
MISSING_MODELS = set()
# Opt-in fallback to a sibling result dir named <model>__<tag> (e.g. a run
# kept aside pending a rerun). Off by default; every use is reported at the end.
USE_QUARANTINED = False
QUARANTINED_USED = set()


def mp(root, model, *rest):
    """Path under <root>/<model>/…, falling back to a quarantined sibling."""
    d = os.path.join(root, model)
    if not os.path.isdir(d) and USE_QUARANTINED:
        alt = sorted(glob.glob(os.path.join(root, model + "__*")))
        if alt:
            QUARANTINED_USED.add(os.path.basename(alt[0]))
            d = alt[0]
    return os.path.join(d, *rest)
NONLLAVA4 = ["Qwen2-VL-7B-Instruct", "smolvlm-2b", "idefics2-8b", "instructblip-vicuna-7b"]
NINE = LLAVA5 + NONLLAVA4
DISPLAY = {
    "llava-1.5-7b": "LLaVA-1.5-7B", "llava-1.5-13b": "LLaVA-1.5-13B",
    "llava-v1.6-vicuna-7b": "LLaVA-v1.6-Vicuna-7B", "llava-v1.6-mistral-7b": "LLaVA-v1.6-Mistral-7B",
    "llama3-llava-next-8b": "Llama3-LLaVA-Next-8B", "Qwen2-VL-7B-Instruct": "Qwen2-VL-7B",
    "smolvlm-2b": "SmolVLM-2B", "idefics2-8b": "IDEFICS2-8B", "instructblip-vicuna-7b": "InstructBLIP",
}
SHORT = {  # panel titles (full names stay in the captions)
    "llava-1.5-7b": "LLaVA-1.5-7B", "llava-1.5-13b": "LLaVA-1.5-13B",
    "llava-v1.6-vicuna-7b": "v1.6-Vicuna-7B", "llava-v1.6-mistral-7b": "v1.6-Mistral-7B",
    "llama3-llava-next-8b": "Llama3-Next-8B", "Qwen2-VL-7B-Instruct": "Qwen2-VL-7B",
    "smolvlm-2b": "SmolVLM-2B", "idefics2-8b": "IDEFICS2-8B", "instructblip-vicuna-7b": "InstructBLIP",
}
MAIN = "llava-1.5-7b"
# The ten Visual Anagram stimuli shown in the Appendix A gallery.
GALLERY_STIMULI = ["va_s003", "va_s031", "va_s036", "va_s047", "va_s048", "va_s049",
                   "va_s055", "va_s056", "va_s057", "va_s091"]
BASE_SLUG = slugify("List every animal in the image, each in one word." + "I see a")
HINT_SLUG = slugify("List every animal in the image, each in one word. There are two animals." + "I see a")

# ---- physical layout (inches) ----
TEXTWIDTH = 6.5      # ACL figure* width
COLWIDTH = 3.0       # ACL column width
DPI = 300

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "legend.title_fontsize": 6.5, "lines.linewidth": 1.3, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.handlelength": 1.1,
    "legend.borderpad": 0.3, "legend.labelspacing": 0.25, "legend.columnspacing": 0.6,
    "legend.handletextpad": 0.5, "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    "figure.dpi": 100,
})

# token colors shared with rotation_first_token / rotation_beam / top_down_cues (COLOR_DICT semantics)
COLOR_DICT = {
    "▁du": "blue", "▁duck": "blue", "Ġduck": "blue",
    "▁bird": "skyblue", "Ġbird": "skyblue",
    "▁rabb": "orange", "▁rab": "orange", "Ġrabbit": "orange",
    "▁b": "salmon", "Ġbunny": "salmon",
}
# beam continuation palettes: exclusive = greens/teals, enumerating = purples.
# Chosen so the within-class colors are distinguishable at print size; a 5th+ curve of a class reuses the colors with a dashed line.
EXCL_COLORS = ["#0b6623", "#2aa198", "#98c93c", "#7fb3d5"]
ENUM_COLORS = ["#8e44ad", "#d33682", "#553c8b", "#e59f71"]
CLASSIFIED_SERIES = (("e_single", "single animal", "#1b7f3b"), ("e_multiple", "multiple animals", "#8e44ad"),
                     ("p_bird", "bird only", "skyblue"), ("p_rabbit", "rabbit only", "orange"),
                     ("p_both", "both percepts", "#c71585"), ("e_none", "no animal", "gray"))
# percept labels (which animal) only mean something on the duck--rabbit stimuli
PERCEPT_KEYS = ("p_bird", "p_rabbit", "p_both")

with open(os.path.join(STIM_DIR, "boundaries.json"), "r", encoding="utf-8") as f:
    BOUNDARIES = json.load(f)


# =============================================================================
# helpers
# =============================================================================
def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[panel] {path}")


def require_nonempty(model, kind, n):
    """A model that contributes zero images would render as a blank panel --
    the aggregate loaders skip missing per-image files (inherited from the
    experiment scripts). Fail loudly instead, so that an empty panel can never
    reach the paper unnoticed. --allow-missing drops the model from the row instead.
    """
    if n == 0:
        raise FileNotFoundError(f"{model}: no {kind} data found (quarantined or not yet rerun)")


def available(models):
    """Filter out models with no rotation cache (quarantined/not yet rerun)."""
    if not ALLOW_MISSING:
        return list(models)
    out = []
    for m in models:
        if os.path.isdir(os.path.dirname(mp(ROT_DIR, m, "x"))):
            out.append(m)
        else:
            MISSING_MODELS.add(m)
    return out


def grid(n, ncols, panel_h, width=TEXTWIDTH, hspace=0.55, wspace=0.35, sharey=False, sharex=False):
    """n panels in ncols columns at the final physical width; unused axes hidden."""
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, nrows * panel_h),
                             squeeze=False, sharey=sharey, sharex=sharex)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.12, hspace=hspace, wspace=wspace)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def valid_va_names(model, slug=BASE_SLUG):
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(STIM_DIR, "va_s*.png")))
    return [n for n in names if get_boundary(BOUNDARIES, model, n, slug) is not None]


def demangle_path(path):
    """Archive-mangled cache names (#U2581 for ▁, #U0120 for Ġ) fall back transparently."""
    if os.path.exists(path):
        return path
    alt = path.replace("▁", "#U2581").replace("Ġ", "#U0120")
    return alt if os.path.exists(alt) else path


def shared_legend(fig, handles, labels, ncol=None, y=None, anchor_axes=None, title=None, pad=0.03):
    """One legend below the whole figure, centered on the given axes span.

    The panels are short, so their tick labels and x-labels are drawn *below*
    the figure box (negative figure coords) and bbox_inches="tight" grows the
    canvas around them. Anchoring the legend at the figure bottom therefore put
    it on top of the x-labels. Measure the drawn extent of
    the axes instead and sit below it; `y` overrides the measurement.
    """
    ncol = ncol or len(labels)
    axes = list(anchor_axes) if anchor_axes is not None else [a for a in fig.axes if a.get_visible()]
    x = 0.5
    if axes:
        pos = [a.get_position() for a in axes]
        x = (min(p.x0 for p in pos) + max(p.x1 for p in pos)) / 2
    if y is None:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        # bottom-most drawn edge of the panels the legend belongs to, labels included
        y = min(a.get_tightbbox(r).transformed(inv).y0 for a in axes) - pad if axes else -0.05
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(x, y), ncol=ncol, title=title)


def panel_legend(ax, ncol=None, y=-0.32, maxlen=16, fontsize=6, **kw):
    """Model-specific legend under its own panel (labels truncated to fit a narrow panel)."""
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    l = [x if len(x) <= maxlen else x[:maxlen - 1] + "…" for x in l]
    if ncol is None:
        ncol = 1 if max(len(x) for x in l) > 9 else 2
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=fontsize, **kw)


def center_xlabel(axes, text):
    """One x-axis label on the middle panel of a row (rows of narrow panels share it)."""
    axes = list(axes)
    axes[len(axes) // 2].set_xlabel(text)


def right_legend(ax, maxlen=16, fontsize=6, **kw):
    h, l = ax.get_legend_handles_labels()
    if h:
        l = [x if len(x) <= maxlen else x[:maxlen - 1] + "…" for x in l]
        ax.legend(h, l, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=fontsize, **kw)


def row_xlabel(fig, axes, text, dy=0.0):
    """One x-axis label centered under a row of panels (instead of one per panel)."""
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    fig.text((x0 + x1) / 2, y0 + dy, text, ha="center", va="top")


def autoscale_y(ax, curves, floor=0.25, lo=None, hi=None, symmetric=False):
    """Fit the y-axis to the plotted band instead of the full [0,1] / [-1,1].

    A fixed full-range axis flattened the cue effects.
    `curves` is a list of (mean, std) arrays; `floor` is the smallest span kept
    so a near-flat panel is not blown up into noise, and lo/hi clamp to the
    quantity's natural range.
    """
    vals = []
    for mean, std in curves:
        mean = np.asarray(mean, dtype=float)
        std = np.zeros_like(mean) if std is None else np.asarray(std, dtype=float)
        vals.append(mean - std)
        vals.append(mean + std)
    v = np.concatenate([a[np.isfinite(a)] for a in vals]) if vals else np.array([0.0, 1.0])
    if v.size == 0:
        return
    vmin, vmax = float(v.min()), float(v.max())
    if symmetric:
        m = max(abs(vmin), abs(vmax), floor / 2)
        vmin, vmax = -m, m
    pad = 0.06 * max(vmax - vmin, 1e-9)
    vmin, vmax = vmin - pad, vmax + pad
    if vmax - vmin < floor:                       # widen a near-flat panel
        c = 0.5 * (vmin + vmax)
        vmin, vmax = c - floor / 2, c + floor / 2
    if lo is not None:
        vmin = max(lo, vmin)
    if hi is not None:
        vmax = min(hi, vmax)
    ax.set_ylim(vmin, vmax)


def tidy(ax, xlabel=None, ylabel=None, title=None, ylim=None, xlim=None, nbins=4):
    if title:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.grid(True)


# =============================================================================
# curve statistics (mirrors utils.plot_topk_tokens, minus the drawing)
# =============================================================================
def curve_stats(runs, centers, color_dict=None, beam_mode=False, max_curves=None):
    per_run, all_angles, order, seen = [], set(), [], set()
    for run, c in zip(runs, centers):
        series = defaultdict(dict)
        for item in sorted(run, key=lambda x: x["angle"]):
            ang = float(item["angle"]) - float(c)
            all_angles.add(ang)
            for tok, p in zip(item.get("tokens", []), item.get("probs", [])):
                series[tok][ang] = p
                if tok not in seen:
                    seen.add(tok)
                    order.append(tok)
        per_run.append(dict(series))
    angles = sorted(all_angles)
    if color_dict:
        ordered = [t for t in color_dict if t in seen]
        order = ordered + [t for t in order if t not in set(ordered)]
    stats = {}
    for tok in order:
        Y = np.full((len(per_run), len(angles)), np.nan)
        for i, rs in enumerate(per_run):
            amap = rs.get(tok, {})
            for j, a in enumerate(angles):
                if a in amap:
                    Y[i, j] = amap[a]
        if np.all(np.isnan(Y)):
            continue
        counts = np.sum(np.isfinite(Y), axis=0)
        mean = np.divide(np.nansum(Y, axis=0), counts, out=np.full(len(angles), np.nan), where=counts > 0)
        diffs = np.where(np.isfinite(Y), Y - mean[None, :], 0.0)
        var = np.divide(np.sum(diffs ** 2, axis=0), counts, out=np.full(len(angles), np.nan), where=counts > 0)
        stats[tok] = (counts, mean, np.sqrt(var))
    order = [t for t in order if t in stats]
    if max_curves is not None and len(order) > max_curves:
        pinned = [t for t in order if color_dict and t in color_dict]
        rest = sorted([t for t in order if t not in set(pinned)], key=lambda t: -float(np.nanmax(stats[t][1])))
        keep = set(pinned) | set(rest[:max(0, max_curves - len(pinned))])
        order = [t for t in order if t in keep]
    # colors / linestyles
    styles, used, n_excl, n_enum = {}, set(), 0, 0
    for tok in order:
        if color_dict and tok in color_dict:
            styles[tok] = (color_dict[tok], "-")
        elif beam_mode:
            if classify_beam_continuation(tok) == "enum":
                styles[tok] = (ENUM_COLORS[n_enum % 4], "-" if n_enum < 4 else "--")
                n_enum += 1
            else:
                styles[tok] = (EXCL_COLORS[n_excl % 4], "-" if n_excl < 4 else "--")
                n_excl += 1
        else:
            styles[tok] = (_stable_color(tok, NEUTRAL_PALETTE, used), "-")
    return angles, [(tok, *stats[tok], *styles[tok]) for tok in order], len(per_run)


def draw_curves(ax, angles, curves, n_runs, xlim=(-45, 45), label_fn=None):
    for tok, counts, mean, std, color, ls in curves:
        if n_runs > 1:
            valid = counts >= 2
            if np.any(valid):
                ax.fill_between(angles, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                where=valid, alpha=0.18, color=color, linewidth=0)
        ax.plot(angles, mean, color=color, linestyle=ls, marker="o" if len(angles) == 1 else None,
                markersize=3, label=label_fn(tok) if label_fn else tok)
    tidy(ax, xlim=xlim)


def beam_label(s):
    """Legend text for a beam continuation: quoted, leading-space marked."""
    return repr(s) if s.strip() else repr(s)


# =============================================================================
# data loaders
# =============================================================================
ROT_DIR = os.path.join(OUTPUTS_DIR, "rotation_first_token", "text")
BEAM_DIR = os.path.join(OUTPUTS_DIR, "rotation_beam", "text")
TD_DIR = os.path.join(OUTPUTS_DIR, "top_down_cues", "text")
# Free-form classes now come from keyword-v2: the vocabulary is the LLM judge's
# own noun table (unprimed extraction from the models' outputs), matched on word
# boundaries, with a "none" class for descriptions asserting no animal. Validated
# at 97.3% agreement against the judge over 227k texts; the
# old substring keyword probs remain at rotation_freeform/probs.
FREE_DIR = os.path.join(OUTPUTS_DIR, "keyword_freeform", "probs")
# single / multiple / no animal partition the generations by how many distinct
# animals are named; bird-only and rabbit-only are subsets of single, and
# both-percepts a subset of multiple -- the same structure the classified beam
# aggregates use (no separate "other animal" class).
FREE_CLASSES = ("single", "duck", "rabbit", "multiple", "both", "none")
RED_DIR = os.path.join(OUTPUTS_DIR, "red_circle", "text")
RB_DIR = os.path.join(OUTPUTS_DIR, "raven_bear", "text")
DLA_DIR = os.path.join(OUTPUTS_DIR, "layerwise_dla")
DPM_DIR = os.path.join(OUTPUTS_DIR, "dominant_patch_map", "cache")
RES_DIR = os.path.join(OUTPUTS_DIR, "resampling_ablation", "plots")
QP_DIR = os.path.join(OUTPUTS_DIR, "query_patching", "tables")
EX_DIR = os.path.join(OUTPUTS_DIR, "exclusivity_patching", "tables")
CLS_PATH = os.path.join(OUTPUTS_DIR, "beam_longcont", "tables", "class_shares.json")
# All models share the 12-token continuation budget. Sole exception: Llama-3 on
# raven_bear, where under the corrected chat template it hedges ("I see a single
# animal in the image, which appears to be ...") and 67% of its beam mass has not
# named an animal within 12 tokens, leaving the share undefined. That stimulus
# alone is taken from its 24-token rerun (single 0.33 -> 1.00); harper and split
# are identical between the budgets, VA single 0.927 -> 0.947.
CLS_OVERRIDE = [("llama3-llava-next-8b", "raven_bear",
                 os.path.join(OUTPUTS_DIR, "beam_longcont_llama3long", "tables", "class_shares.json"))]


def load_rotation(model, dataset):
    """rotation_first_token: harper single run; va_all / split_all = runs + centers (rotation_first_token __main__ logic)."""
    if dataset == "harper":
        p = mp(ROT_DIR, model, "harper", f"{BASE_SLUG}.json")
        return [extract_topk_from_file(p, k=3, extra_tokens=list(COLOR_DICT))], [0]
    runs, centers = [], []
    for va in valid_va_names(model):
        name = va if dataset == "va_all" else va.replace("va_", "split_", 1)
        p = mp(ROT_DIR, model, name, f"{BASE_SLUG}.json")
        if not os.path.exists(p):
            continue
        runs.append(extract_topk_from_file(p, k=3, extra_tokens=list(COLOR_DICT)))
        centers.append(get_boundary(BOUNDARIES, model, va, BASE_SLUG) if dataset == "va_all" else 0)
    require_nonempty(model, f"rotation/{dataset}", len(runs))
    return runs, centers


def center_strings(results, center):
    entry = next((it["output"][0] for it in results if it.get("angle") == center and it.get("output")), None)
    if entry is None:
        return []
    out, seen = [], set()
    for s in entry.get("tokens", []):
        if isinstance(s, str) and s.strip() and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_beam_single(path, center=0, center_path=None):
    """rotation_beam / raven_bear per-image beam plot input: top-5 at the center + the center beams."""
    with open(center_path or path, "r", encoding="utf-8") as f:
        cs = center_strings(json.load(f), 0 if center_path else center)
    return extract_topk_from_file(path, k=5, extra_tokens=cs, anchor_angle=center)


TD_CUES = {
    "prefix": ["b", "bi", "bu", "r", "d", "x"],
    "semantic": ["avian", "an egg-layer", "nesting", "singing", "terrestrial", "viviparous", "herbivorous", "Easter"],
}
TD_CUES["prefix_negation"] = TD_CUES["prefix"]
TD_CUES["semantic_negation"] = TD_CUES["semantic"]
# Axis labels: the tick labels fill the slot, so the placeholder is an ellipsis
# rather than a literal "{cue}".
TD_XLABEL = {"prefix": "It starts with '\u2026'.", "semantic": "It's \u2026.",
             "prefix_negation": "It doesn't start with '\u2026'.", "semantic_negation": "It's not \u2026."}
TD_BASE = "List every animal in the image, each in one word."


def td_prompt(cue_type, cue):
    if cue == "no cue":
        return TD_BASE
    return TD_BASE + {"prefix": f" It starts with '{cue}'.", "semantic": f" It's {cue}.",
                      "prefix_negation": f" It doesn't start with '{cue}'.",
                      "semantic_negation": f" It's not {cue}."}[cue_type]


def load_topdown(model, dataset, cue_type):
    """top_down_cues plot input: list over images of {cue_label: {'prob_data', 'center_angle'}}."""
    if dataset == "harper":
        images = [("harper", 0)]
    else:
        prefix = {"va_all": "va_", "split_all": "split_", "duck_all": "duck_", "rabbit_all": "rabbit_"}[dataset]
        images = [(va.replace("va_", prefix, 1), get_boundary(BOUNDARIES, model, va, BASE_SLUG) if dataset == "va_all" else 0)
                  for va in valid_va_names(model)]
    records = []
    for name, center in images:
        rec = {}
        for cue in ["no cue"] + TD_CUES[cue_type]:
            p = mp(TD_DIR, model, name, f"{slugify(td_prompt(cue_type, cue) + 'I see a')}.json")
            if not os.path.exists(p):
                continue
            rec[cue] = {"prob_data": extract_topk_from_file(p, k=5, extra_tokens=list(COLOR_DICT), anchor_angle=center),
                        "center_angle": center}
        if rec:
            records.append(rec)
    require_nonempty(model, f"top-down/{dataset}/{cue_type}", len(records))
    return records


def topdown_stats(records, cue_type):
    """Mean/std across samples (image x relative angle) per token per cue (plot_cue_effects_va_all)."""
    x_labels = ["no cue"] + TD_CUES[cue_type]
    samples = []
    for rec in records:
        maps, rels = {}, set()
        for cue, r in rec.items():
            maps[cue] = build_token_angle_map(r["prob_data"], r["center_angle"])
            rels.update(maps[cue].keys())
        for rel in sorted(rels):
            samples.append({cue: maps.get(cue, {}).get(rel, {}) for cue in maps})
    out = {}
    for tok in COLOR_DICT:
        Y = np.array([[s.get(c, {}).get(tok, np.nan) for c in x_labels] for s in samples], dtype=float)
        if Y.size == 0 or np.all(np.isnan(Y)):
            continue
        counts = np.sum(np.isfinite(Y), axis=0)
        mean = np.divide(np.nansum(Y, axis=0), counts, out=np.full(len(x_labels), np.nan), where=counts > 0)
        diffs = np.where(np.isfinite(Y), Y - mean[None, :], 0.0)
        std = np.sqrt(np.divide(np.sum(diffs ** 2, axis=0), counts, out=np.full(len(x_labels), np.nan), where=counts > 0))
        out[tok] = (counts, mean, std)
    return x_labels, out


def draw_topdown(ax, x_labels, stats, ylim="auto", rotation=40, tick_fontsize=None):
    x = np.arange(len(x_labels))
    for tok, (counts, mean, std) in stats.items():
        color = COLOR_DICT[tok]
        valid = counts >= 2
        if np.any(valid):
            ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), where=valid,
                            alpha=0.15, color=color, linewidth=0)
        ax.plot(x, mean, marker="o", markersize=2.5, color=color, label=tok)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotation, ha="right" if rotation not in (0, 90) else "center",
                       fontsize=tick_fontsize)
    tidy(ax, ylim=None if ylim == "auto" else ylim)
    if ylim == "auto":
        autoscale_y(ax, [(m, sd) for _, (_, m, sd) in stats.items()], floor=0.25, lo=0.0, hi=1.0)


# Boundary analyses. The claim is that a cue raises
# the cued percept only where the image supports it, so the figure tracks the
# *cued percept's own mass* across three levels of visual evidence for it:
# none (the image shows the other animal), ambiguous, full (the image shows the
# cued animal). A duck-minus-rabbit preference cannot show this: on a rabbit-only
# image the 'd' cue does move the preference, but by suppressing the rabbit
# tokens, not by producing duck ones, and the freed mass goes to other d-animals
# (deer/dog/donkey) that are not in the image. Split controls are deliberately
# not used here: with both animals visually available, cue-driven
# selection is expected and says nothing about overriding the image.
TD_DUCK_SIDE = tuple(t for t, c in COLOR_DICT.items() if c in ("blue", "skyblue"))
TD_RABBIT_SIDE = tuple(t for t, c in COLOR_DICT.items() if c in ("orange", "salmon"))
TD_NEGATION_COLOR = "#7b3294"   # ambiguous-stimulus purple
# (cue value, cued side, its tokens, color, evidence datasets ordered none -> full).
# The semantic entries use the strongest cue for each side -- "avian" and the
# paper's flagship "Easter" cue (the one with the human precedent). Averaging all
# eight semantic cues would dilute the ladder: the narrower property cues
# (nesting, singing, egg-laying) barely move anything even on ambiguous stimuli,
# and "terrestrial" is a weak rabbit cue in several models (see the negation
# figures for the per-cue picture).
# (cues, cued side, its tokens, colour, evidence datasets ordered none -> ambiguous).
# The "full" level (the image of the cued animal) is not plotted:
# it is not a boundary condition, only the model naming what it sees. Semantic
# entries average ALL four cues on each side rather than the strongest one
# -- averaging shrinks the ambiguous-level effect but leaves
# the no-evidence level unchanged, which is the contrast the figure is for.
TD_EVIDENCE_CUES = {
    "prefix": ((("d",), "duck", TD_DUCK_SIDE, "blue", ("rabbit_all", "va_all")),
               (("r",), "rabbit", TD_RABBIT_SIDE, "orange", ("duck_all", "va_all"))),
    "semantic": ((("avian", "an egg-layer", "nesting", "singing"), "duck", TD_DUCK_SIDE, "blue",
                  ("rabbit_all", "va_all")),
                 (("terrestrial", "viviparous", "herbivorous", "Easter"), "rabbit", TD_RABBIT_SIDE, "orange",
                  ("duck_all", "va_all"))),
}
TD_EVIDENCE_LEVELS = ("none", "ambiguous")


def _side_mass(probs, tokens):
    """Summed probability of one side's tokens.

    extract_topk_from_file stores NaN (not absence) for a requested token that
    is outside the file's top-k, and COLOR_DICT lists every tokenizer's spelling
    of the four words, so most entries are NaN for any one model. Treat those as
    zero mass -- summing them naively makes the whole side NaN.
    """
    total = 0.0
    for t in tokens:
        v = probs.get(t)
        if v is not None and np.isfinite(v):
            total += float(v)
    return total


def _topdown_samples(records):
    """One sample per stimulus (x relative angle): {cue: {token: prob}}."""
    samples = []
    for rec in records:
        maps, rels = {}, set()
        for cue, r in rec.items():
            maps[cue] = build_token_angle_map(r["prob_data"], r["center_angle"])
            rels.update(maps[cue].keys())
        for rel in sorted(rels):
            samples.append({cue: maps.get(cue, {}).get(rel, {}) for cue in maps})
    return samples


def topdown_side_stats(records, cue_type, cue_label, side_tokens):
    """Mean/SD of one side's probability mass under a single cue."""
    Y = np.array([_side_mass(s[cue_label], side_tokens) for s in _topdown_samples(records)
                  if s.get(cue_label)], dtype=float)
    if Y.size == 0:
        return 0, (np.nan, np.nan)
    return len(Y), (float(Y.mean()), float(Y.std()))


def topdown_pref_stats(records, cue_type, relative=False):
    """Mean/SD of (duck-side - rabbit-side) probability mass per cue.

    A token absent from the stored top-k contributes 0 to its side (it is
    below the 100th rank, i.e. negligible mass); a cue whose run is missing
    for a stimulus contributes nothing to that cue's mean. With relative=True
    each stimulus is expressed as a paired difference from its own no-cue run,
    which is the quantity the negation figures compare across cue polarities.
    """
    x_labels = ["no cue"] + TD_CUES[cue_type]
    samples = _topdown_samples(records)
    Y = np.full((len(samples), len(x_labels)), np.nan)
    for i, s in enumerate(samples):
        for j, cue in enumerate(x_labels):
            probs = s.get(cue)
            if probs:
                Y[i, j] = _side_mass(probs, TD_DUCK_SIDE) - _side_mass(probs, TD_RABBIT_SIDE)
    if relative and Y.size:
        Y = Y - Y[:, [0]]
    counts = np.sum(np.isfinite(Y), axis=0)
    mean = np.divide(np.nansum(Y, axis=0), counts, out=np.full(len(x_labels), np.nan), where=counts > 0)
    diffs = np.where(np.isfinite(Y), Y - mean[None, :], 0.0)
    std = np.sqrt(np.divide(np.sum(diffs ** 2, axis=0), counts, out=np.full(len(x_labels), np.nan), where=counts > 0))
    return x_labels, (counts, mean, std)


def draw_topdown_pref(ax, x_labels, series, ylim="auto", tick_fontsize=5.5):
    x = np.arange(len(x_labels))
    clip = (-1, 1) if ylim == "auto" else ylim
    ax.axhline(0.0, color="0.35", linewidth=0.6, zorder=1)
    for label, color, (counts, mean, std) in series:
        valid = counts >= 2
        if np.any(valid):
            ax.fill_between(x, np.clip(mean - std, *clip), np.clip(mean + std, *clip), where=valid,
                            alpha=0.15, color=color, linewidth=0)
        ax.plot(x, mean, marker="o", markersize=2.5, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=tick_fontsize)
    tidy(ax, ylim=None if ylim == "auto" else ylim)
    if ylim == "auto":
        autoscale_y(ax, [(m, sd) for _, _, (_, m, sd) in series], floor=0.3, lo=-1.0, hi=1.0, symmetric=True)


FREE_ANGLE_LIM = 45


def load_free(model, dataset, slug):
    """rotation_freeform keyword-class probabilities; va_all aggregated by relative angle."""
    # The rotation_freeform runs sweep +-60 deg; the paper's figures use +-45 like every
    # other rotation figure.
    if dataset == "harper":
        with open(mp(FREE_DIR, model, "harper", f"{slug}.json")) as f:
            d = [it for it in json.load(f) if abs(it["angle"]) <= FREE_ANGLE_LIM]
        rel = [it["angle"] for it in d]
        return rel, {k: (np.array([it[k] for it in d]), None) for k in FREE_CLASSES}
    acc = defaultdict(lambda: defaultdict(list))
    for va in valid_va_names(model):
        p = mp(FREE_DIR, model, va, f"{slug}.json")
        if not os.path.exists(p):
            continue
        c = get_boundary(BOUNDARIES, model, va, BASE_SLUG)
        with open(p) as f:
            for it in json.load(f):
                for k in FREE_CLASSES:
                    if abs(it["angle"] - c) <= FREE_ANGLE_LIM:
                        acc[int(it["angle"] - c)][k].append(float(it[k]))
    require_nonempty(model, f"free-form/{slug}/{dataset}", len(acc))
    rel = sorted(acc)
    return rel, {k: (np.array([np.mean(acc[r][k]) for r in rel]), np.array([np.std(acc[r][k]) for r in rel]))
                 for k in FREE_CLASSES}


# Same wording and colours as the classified beam aggregates (CLASSIFIED_SERIES)
# so the two views of "what did it report" read as one family.
# The classes are the free-form ones (a report can name a third animal, which
# beam search never separates out), hence the extra "other animal".
FREE_LABELS = {"single": "single animal", "duck": "bird only", "rabbit": "rabbit only",
               "multiple": "multiple animals", "both": "both percepts", "none": "no animal"}
FREE_COLORS = {"single": "#1b7f3b", "duck": "skyblue", "rabbit": "orange",
               "multiple": "#8e44ad", "both": "#c71585", "none": "gray"}


def draw_free(ax, rel, stats):
    for k, (mean, std) in stats.items():
        if std is not None:
            ax.fill_between(rel, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.18,
                            color=FREE_COLORS[k], linewidth=0)
        ax.plot(rel, mean, color=FREE_COLORS[k], label=FREE_LABELS[k])
    tidy(ax, ylim=(0, 1))


def load_classified():
    with open(CLS_PATH) as f:
        curves = json.load(f)["curves"]
    for model, cls, path in CLS_OVERRIDE:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            src = json.load(f)["curves"]
        for key, rows in src.items():
            m, _, c = key.split("|")
            if m == model and c == cls:
                curves[key] = rows
    return curves


# raven_bear percepts (from classify_beam_longcont): p_other =
# the mammal reading only (bear / dog / panda ...), p_both = bird + mammal.
RAVEN_SERIES = (("e_single", "single animal", "#1b7f3b"), ("e_multiple", "multiple animals", "#8e44ad"),
                ("p_bird", "bird only", "skyblue"), ("p_other", "bear/dog only", "orange"),
                ("p_both", "both percepts", "#c71585"), ("e_none", "no animal", "gray"))


def draw_classified(ax, rows, cls):
    x = [r["rel_angle"] for r in rows]
    for k, label, col in (RAVEN_SERIES if cls == "raven_bear" else CLASSIFIED_SERIES):
        if k in PERCEPT_KEYS and cls not in ("va", "split", "harper", "raven_bear"):
            continue
        y = np.array([r[k] for r in rows])
        sd = np.array([r[k + "_std"] for r in rows])
        if len(rows) > 1 and rows[0]["n_images"] > 1:
            ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), alpha=0.18, color=col, linewidth=0)
        ax.plot(x, y, color=col, label=label, marker="o" if len(x) == 1 else None)
    tidy(ax, ylim=(0, 1), xlim=(min(x), max(x)))


def red_circle_diff_map(model, image="harper"):
    """(base RGB array, diff heat map, vmax) -- the Δ(rabbit − bird) panel of utils.plot_heatmaps."""
    bird, rabb = get_duck_rabbit_tokens(model)
    p = demangle_path(mp(RED_DIR, model, image, f"{BASE_SLUG}_{bird}_{rabb}_r0p071_s0p143_lw4.json"))
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    centers, g1, g2, b1, b2 = [], [], [], None, None
    for it in data:
        c = it.get("center", "missing")
        outs = it.get("output", [])
        if c == "missing" or not outs:
            continue
        o = outs[0]
        tv = dict(zip(o["tokens"], o["logits"] if len(o.get("logits", [])) == len(o["tokens"]) else o["probs"]))
        v1, v2 = tv.get(rabb, np.nan), tv.get(bird, np.nan)  # group1 = rabbit, group2 = bird (red_circle)
        if c is None:
            b1, b2 = v1, v2
            continue
        centers.append(c)
        g1.append(v1)
        g2.append(v2)
    centers = np.array(centers, dtype=float)
    g1, g2 = np.array(g1), np.array(g2)
    diff = (g1 - g2) - (b1 - b2)
    # the experiments used the 425x425 white-composited canvas (RotatedRGBAImageDataset at 0°, no resize)
    base = RotatedRGBAImageDataset(os.path.join(STIM_DIR, f"{image}.png"), [0], output_size=None)[0]
    w, h = base.size
    xs, ys = centers[:, 0].astype(int), centers[:, 1].astype(int)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, diff = xs[inside], ys[inside], diff[inside]
    xu, yu = np.sort(np.unique(xs)), np.sort(np.unique(ys))
    gridv = np.full((len(yu), len(xu)), np.nan)
    xi, yi = {x: i for i, x in enumerate(xu)}, {y: i for i, y in enumerate(yu)}
    for x, y, v in zip(xs, ys, diff):
        gridv[yi[y], xi[x]] = v
    heat = _upsample_grid_bilinear(gridv, xu, yu, w, h)
    valid = heat[~np.isnan(heat)]
    vmax = float(np.max(np.abs(valid))) if valid.size else 1.0
    return np.array(base), heat, max(vmax, 1e-6)


def draw_heat(ax, base, heat, vmax, cmap="RdBu_r", alpha=0.6):
    ax.imshow(base)
    im = ax.imshow(np.ma.masked_invalid(heat), alpha=alpha, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


HUMAN_MAP_PATH = os.path.join(FIGURES_DIR, "external", "hsu_chen_duck_rabbit.png")


def human_map():
    """Right panel (Rabbit − Duck) of the Hsu & Chen (2025) fixation-density
    figure (LaTeX trim 1020bp ≈ px 1380). The figure is third-party material and
    is not redistributed with this repository: place it at HUMAN_MAP_PATH to
    reproduce the paper's panel; otherwise the panel is left empty."""
    if not os.path.exists(HUMAN_MAP_PATH):
        print(f"[human_map] {HUMAN_MAP_PATH} not found; the human-fixation panel is left empty")
        return None
    im = Image.open(HUMAN_MAP_PATH).convert("RGB")
    return np.array(im.crop((1380, 0, im.width, im.height)))


def draw_human(ax, title):
    hm = human_map()
    if hm is not None:
        ax.imshow(hm)
    else:
        ax.text(0.5, 0.5, "human fixation map\n(not redistributed)", ha="center", va="center",
                fontsize=5, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title)


def load_dla(model):
    slug_dir = mp(os.path.join(DLA_DIR, "plots", "layerwise_dla"), model, BASE_SLUG)
    with open(os.path.join(slug_dir, "included_images.json")) as f:
        images = json.load(f)["images"]
    curves = []
    for im in images:
        p = mp(os.path.join(DLA_DIR, "cache"), model, im, BASE_SLUG, "layerwise_dla_last_bird_rabb.npz")
        if os.path.exists(p):
            d = np.load(p)
            curves.append({k: d[k] for k in ("resid_post", "attn_out", "mlp_out", "attn_out_image_only")})
    return curves


DLA_COMPONENTS = [("resid_post", "resid_post", "-", 1.6), ("attn_out", "attn_out", ":", 1.6),
                  ("attn_out_image_only", "attn_out (image only)", ":", 1.2), ("mlp_out", "mlp_out", "--", 1.4)]
DLA_COLORS = {"resid_post": ("tab:blue", "tab:orange"), "attn_out": ("tab:blue", "tab:orange"),
              "attn_out_image_only": ("deepskyblue", "red"), "mlp_out": ("royalblue", "darkorange")}


def draw_dla(ax, curves, token_labels):
    L = curves[0]["resid_post"].shape[0]
    layers = np.arange(L)
    for key, label, ls, lw in DLA_COMPONENTS:
        arr = np.stack([c[key] for c in curves], axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        for ti, tok in enumerate(token_labels):
            color = DLA_COLORS[key][ti]
            ax.plot(layers, mean[:, ti], linestyle=ls, linewidth=lw, color=color, label=f"{label} | {tok}")
            ax.fill_between(layers, mean[:, ti] - std[:, ti], mean[:, ti] + std[:, ti], color=color, alpha=0.08, linewidth=0)
    ax.axhline(0.0, linewidth=0.6, color="gray")
    ax.set_xlim(0, L - 1)
    ax.set_xticks(np.arange(0, L, 10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, axis="y")


def load_dpm(model):
    with open(mp(DPM_DIR, model, f"{BASE_SLUG}__records.json")) as f:
        return json.load(f)


def draw_dpm(ax, records, vmax=None):
    diffs = np.array([r["score_diff"] for r in records if np.isfinite(r["score_diff"])])
    if vmax is None:
        vmax = max(float(np.nanmax(np.abs(diffs))) if len(diffs) else 1.0, 1e-6)
    sc = None
    for cond, marker, kw in (("rotation", "x", dict(linewidths=0.8)), ("red_circle", "o", dict(linewidths=0.6))):
        sub = [r for r in records if r["condition"] == cond and np.isfinite(r["score_diff"])]
        if sub:
            sc = ax.scatter([r["n_dom_a"] for r in sub], [r["n_dom_b"] for r in sub], c=[r["score_diff"] for r in sub],
                            cmap="coolwarm_r", vmin=-vmax, vmax=vmax, s=12, alpha=0.9, marker=marker, label=cond, **kw)
    tidy(ax)
    return sc


def load_query_patching(model):
    lab = "0-32" if model == "llava-1.5-13b" else "0-31"
    d = mp(QP_DIR, model, "combined_bottomup_vs_topdown")
    with open(os.path.join(d, f"query_gen__layers_{lab}.json")) as f:
        recs = json.load(f)
    with open(os.path.join(d, f"query_gen__layers_{lab}__fit.json")) as f:
        fit = json.load(f)
    return recs, fit


def draw_query_patching(ax, recs, fit, annotate=True):
    fin = lambda r: np.isfinite(r["x"]) and np.isfinite(r["y"])
    rot = [r for r in recs if r["condition"] == "rotation" and fin(r)]
    red = [r for r in recs if r["condition"] == "red_circle" and fin(r)]
    td = [r for r in recs if r["condition"] == "top_down" and fin(r)]
    if rot:
        ax.scatter([r["x"] for r in rot], [r["y"] for r in rot], marker="x", s=4, linewidths=0.4, alpha=0.8,
                   color="black", label="bottom-up: rotation", zorder=2)
    if red:
        ax.scatter([r["x"] for r in red], [r["y"] for r in red], marker=".", s=4, alpha=0.95, color="black",
                   label="bottom-up: red circle", zorder=3)
    if td:
        ax.scatter([r["x"] for r in td], [r["y"] for r in td], marker="s", s=4, alpha=0.85, color="tab:green",
                   label="top-down: x→r/d", zorder=2)
    ax.axhline(0, linestyle=":", linewidth=0.6, alpha=0.6, color="gray")
    ax.axvline(0, linestyle=":", linewidth=0.6, alpha=0.6, color="gray")
    xs = np.array([r["x"] for r in recs if np.isfinite(r["x"])])
    ys = np.array([r["y"] for r in recs if np.isfinite(r["y"])])
    lim = float(max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1e-6)) if len(xs) and len(ys) else 1.0
    xx = np.array([-lim * 1.05, lim * 1.05])
    bs, ts = fit["bottom_up"]["slope"], fit["top_down"]["slope"]
    if np.isfinite(bs):
        ax.plot(xx, bs * xx, color="black", linewidth=1.2, label="bottom-up fit")
    if np.isfinite(ts):
        ax.plot(xx, ts * xx, color="darkgreen", linewidth=1.2, linestyle="--", label="top-down fit")
    if annotate:
        ax.text(0.03, 0.97, f"β(bottom-up) = {bs:.3f}\nβ(top-down) = {ts:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    tidy(ax, xlim=(-lim * 1.05, lim * 1.05), ylim=(-lim * 1.05, lim * 1.05))


def load_exclusivity(model):
    with open(mp(EX_DIR, model, "records.json")) as f:
        return json.load(f)


def draw_exclusivity(ax, records, jitter_frac=0.003, seed=0, s_dot=1.2, s_star=4.0):
    xs_all = np.array([r["x"] for r in records if np.isfinite(r["x"])])
    ys_all = np.array([r["y"] for r in records if np.isfinite(r["y"])])
    xj = max(float(np.ptp(xs_all)) if len(xs_all) else 1.0, 1e-6) * jitter_frac
    yj = max(float(np.ptp(ys_all)) if len(ys_all) else 1.0, 1e-6) * jitter_frac
    rng = np.random.default_rng(seed)
    cmap = {"bird": "tab:blue", "rabbit": "tab:orange", "none": "black"}
    for cl, ml in [("bird", "none"), ("rabbit", "none"), ("none", "none"), ("bird", "two"), ("rabbit", "two"), ("none", "two")]:
        sub = [r for r in records if r["color_label"] == cl and r["marker_label"] == ml and np.isfinite(r["x"]) and np.isfinite(r["y"])]
        if not sub:
            continue
        xs = np.array([r["x"] for r in sub]) + rng.normal(0, xj, len(sub))
        ys = np.array([r["y"] for r in sub]) + rng.normal(0, yj, len(sub))
        if ml == "two":
            ax.scatter(xs, ys, marker="*", c="red", s=s_star, alpha=0.9, linewidths=0.2, zorder=3)
        else:
            ax.scatter(xs, ys, marker="o", c=cmap[cl], s=s_dot, alpha=0.65, linewidths=0, zorder=2)
    ax.axhline(0, linestyle=":", linewidth=0.6, alpha=0.6, color="gray")
    ax.axvline(0, linestyle=":", linewidth=0.6, alpha=0.6, color="gray")
    tidy(ax)


def draw_exclusivity_joint(fig, records, gs=None, xlabel=None, ylabel=None, hist_frac=0.22,
                           label_counts=True):
    """Scatter with marginal histograms.

    The point of the figure is that *most* single-token patches move neither
    measure; the marginals make that mass at ~0 explicit instead of leaving it
    to overplotting. Counts are on a log axis so the tail
    that does matter stays visible next to the spike at zero.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    spec = dict(width_ratios=[1 - hist_frac, hist_frac], height_ratios=[hist_frac, 1 - hist_frac],
                wspace=0.04, hspace=0.04)
    g = (GridSpecFromSubplotSpec(2, 2, subplot_spec=gs, **spec) if gs is not None
         else fig.add_gridspec(2, 2, **spec))
    ax = fig.add_subplot(g[1, 0])
    ax_top = fig.add_subplot(g[0, 0], sharex=ax)
    ax_right = fig.add_subplot(g[1, 1], sharey=ax)

    draw_exclusivity(ax, records)
    finite = [r for r in records if np.isfinite(r["x"]) and np.isfinite(r["y"])]
    xs = np.array([r["x"] for r in finite], dtype=float)
    ys = np.array([r["y"] for r in finite], dtype=float)
    two = np.array([r["marker_label"] == "two" for r in finite], dtype=bool)
    # grey = every patched token (incl. the bird-/rabbit-dominant ones);
    # red = the '"two"-dominant' subset drawn as stars in the scatter. Shared
    # bin edges, or the overlay would not align with the bars it sits in.
    for axis, vals, orient in ((ax_top, xs, "vertical"), (ax_right, ys, "horizontal")):
        edges = np.histogram_bin_edges(vals, bins=60)
        axis.hist(vals, bins=edges, orientation=orient, color="0.55", linewidth=0)
        # every subset as an outline over the grey total: filled bars hid each
        # other, and the three are then read the same way
        subsets = [(two, "red")]
        for lbl, colr in (("bird", "tab:blue"), ("rabbit", "tab:orange")):
            subsets.append((np.array([r["color_label"] == lbl for r in finite], dtype=bool), colr))
        for m, colr in subsets:
            if m.any():
                axis.hist(vals[m], bins=edges, orientation=orient, histtype="step",
                          color=colr, linewidth=0.5)
        if orient == "vertical":
            axis.set_yscale("log")
            axis.tick_params(labelbottom=False, labelsize=5, length=2, width=0.4)
        else:
            axis.set_xscale("log")
            axis.tick_params(labelleft=False, labelsize=5, length=2, width=0.4)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.grid(True, alpha=0.25)
    if label_counts:
        ax_top.set_ylabel("count", fontsize=5.5)
        ax_right.set_xlabel("count", fontsize=5.5)
    else:                                   # a row of panels has no room for them
        ax_top.tick_params(labelleft=False)
        ax_right.tick_params(labelbottom=False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax, ax_top, ax_right


EXCL_LEGEND = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markeredgecolor="tab:blue", markersize=5, label="bird-dominant"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markeredgecolor="tab:orange", markersize=5, label="rabbit-dominant"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markeredgecolor="black", markersize=5, label="below bird/rabbit threshold"),
    Line2D([0], [0], marker="*", color="red", linestyle="None", markersize=7, label='"two" dominant'),
]


# ---- resampling heatmaps (needs a processor for the post-image token labels; aggregate_resampling logic) ----
RES_PROMPT_FORMATS = {
    "llava-1.5-7b": "USER: <image>\n{prompt} ASSISTANT:",
    "llava-1.5-13b": "USER: <image>\n{prompt} ASSISTANT:",
    "llava-v1.6-vicuna-7b": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:",
    "llava-v1.6-mistral-7b": "[INST] <image>\n{prompt} [/INST]",
    "llama3-llava-next-8b": "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
}
RES_X = slugify("List every animal in the image, each in one word. It starts with 'x'." + "It is a")
RES_D = slugify("List every animal in the image, each in one word. It starts with 'd'." + "It is a")
_RES_LABELS = {}


def resampling_labels(model):
    """Post-image text-token labels (decoded), replicated from aggregate_resampling.py."""
    if model in _RES_LABELS:
        return _RES_LABELS[model]
    from collections import Counter
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(VLM_DICT[model])
    va = valid_va_names(model)[0]
    img = centered_va_image(os.path.join(STIM_DIR, f"{va}.png"), BOUNDARIES, model, va, BASE_SLUG)
    prompt = RES_PROMPT_FORMATS[model].format(prompt="List every animal in the image, each in one word. It starts with 'd'.") + "It is a"
    ids = processor(text=prompt, images=img, return_tensors="pt")["input_ids"][0].tolist()
    img_tok = Counter(ids).most_common(1)[0][0]
    last_img = max(i for i, t in enumerate(ids) if t == img_tok)
    n_post = len(ids) - (last_img + 1)
    labels = [processor.tokenizer.decode([t]) for t in ids[-n_post:]]
    _RES_LABELS[model] = labels
    return labels


def load_resampling(model, component):
    d = mp(os.path.join(RES_DIR, component, "text_tokens"), model, "va_all_va_all", f"{RES_X}_{RES_D}")
    store, orig = np.load(os.path.join(d, "mean_store.npy")), np.load(os.path.join(d, "mean_original.npy"))
    labels = resampling_labels(model)
    n_post = len(labels)
    ld = store[:, -n_post:, :] - orig[None, None, :]     # (L, T, tokens); index 0 = duck token, 1 = rabbit token
    return (ld[:, :, 0] - ld[:, :, 1]).T, labels         # (T, L)


def draw_resampling(ax, mat, labels, component, ylabels=True):
    amax = float(np.max(np.abs(mat)))
    im = ax.imshow(mat, cmap="RdBu", vmin=-amax, vmax=amax, aspect="auto", interpolation="nearest")
    L = mat.shape[1]
    if component == "resid_post":
        ticks = [0] + (np.arange(10, L, 10) + 1).tolist()
        ax.set_xticks(ticks)
        ax.set_xticklabels(["pre"] + np.arange(10, L, 10).tolist())
    else:
        ax.set_xticks(np.arange(0, L, 10))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([repr(l)[1:-1] for l in labels] if ylabels else [], fontsize=5.5)
    ax.tick_params(axis="y", length=1.5)
    return im


def slim_colorbar(fig, im, ax, label=None, location="bottom"):
    cb = fig.colorbar(im, ax=ax, location=location, fraction=0.05, pad=0.08 if location == "bottom" else 0.03, shrink=0.9)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    if label:
        cb.set_label(label, fontsize=6)
    return cb


# =============================================================================
# figures
# =============================================================================
FIGS = {}


def figure(name):
    def deco(fn):
        FIGS[name] = fn
        return fn
    return deco


def rotation_row(name, models, dataset, width=TEXTWIDTH):
    models = available(models)
    fig, axes = grid(len(models), len(models), 1.25, width=width, hspace=0.6, wspace=0.4)
    for ax, m in zip(axes, models):
        runs, centers = load_rotation(m, dataset)
        angles, curves, n = curve_stats(runs, centers, COLOR_DICT, max_curves=8)
        draw_curves(ax, angles, curves, n)
        tidy(ax, xlabel="Angle", title=SHORT[m])
        panel_legend(ax, y=-0.42)
    axes[0].set_ylabel("Probability")
    savefig(fig, name)


def beam_row(name, models, prefix, width=TEXTWIDTH):
    models = available(models)
    slug = f"{BASE_SLUG}-{prefix}"
    fig, axes = grid(len(models), len(models), 1.25, width=width, hspace=0.6, wspace=0.4)
    for ax, m in zip(axes, models):
        prob = load_beam_single(mp(BEAM_DIR, m, "harper", f"{slug}.json"), center=0)
        angles, curves, n = curve_stats([prob], [0], COLOR_DICT, beam_mode=True, max_curves=8)
        draw_curves(ax, angles, curves, n, label_fn=beam_label)
        tidy(ax, xlabel="Angle", title=SHORT[m])
        panel_legend(ax, ncol=1, y=-0.42)
    axes[0].set_ylabel("Probability")
    savefig(fig, name)


def classified_row(name, models, slug, cls, ncols=5):
    models = available(models)
    curves = load_classified()
    fig, axes = grid(len(models), ncols, 1.2, hspace=0.75, wspace=0.4)
    for ax, m in zip(axes, models):
        rows = curves.get(f"{m}|{slug}|{cls}")
        if not rows:
            if not ALLOW_MISSING:
                raise FileNotFoundError(f"{m}: no classified-beam curves for {slug}|{cls}")
            MISSING_MODELS.add(m)
            ax.set_visible(False)
            continue
        draw_classified(ax, rows, cls)
        tidy(ax, xlabel="Angle", title=SHORT[m])
    for ax in axes[::ncols]:
        ax.set_ylabel("Share of beam mass")
    h, l = axes[0].get_legend_handles_labels()
    shared_legend(fig, h, l, anchor_axes=list(axes))
    savefig(fig, name)


def topdown_row(name, models, dataset, cue_type, ncols=5, width=TEXTWIDTH, ylim="auto"):
    models = available(models)
    prefix = cue_type.startswith("prefix")
    fig, axes = grid(len(models), ncols, 1.25, width=width, hspace=1.5 if prefix else 2.0, wspace=0.4)
    for ax, m in zip(axes, models):
        x_labels, stats = topdown_stats(load_topdown(m, dataset, cue_type), cue_type)
        draw_topdown(ax, x_labels, stats, ylim=ylim, rotation=40, tick_fontsize=5.5)
        tidy(ax, title=SHORT[m])
        panel_legend(ax, ncol=2, y=-0.55 if prefix else -0.95)
    for ax in axes[::ncols]:
        ax.set_ylabel("Probability")
    for r in range(math.ceil(len(models) / ncols)):
        center_xlabel(axes[r * ncols:(r + 1) * ncols], TD_XLABEL[cue_type])
    savefig(fig, name)


def topdown_evidence_row(name, models, cue_type, ncols=5, width=TEXTWIDTH):
    """The cued percept's own mass against the visual evidence available for it.

    For each cue that names an interpretation ('d'/'r', or the semantic cues of
    each side, averaged), the cued side's probability mass is read off two stimulus
    sets: the image of the other animal (no evidence) and the ambiguous stimulus.
    Solid = with the cue, dashed = the same stimuli with no cue, so both the level
    and the increment are visible.
    """
    models = available(models)
    # hspace leaves room for the per-row x label; with only two short categories
    # the tick labels stay horizontal (rotating them collided with the next row's
    # titles at ncols=5).
    fig, axes = grid(len(models), ncols, 1.25 if ncols >= 5 else 1.7, width=width,
                     hspace=1.15, wspace=0.35)
    x = np.arange(len(TD_EVIDENCE_LEVELS))
    for ax, m in zip(axes, models):
        for cues, side_name, side_tokens, color, datasets in TD_EVIDENCE_CUES[cue_type]:
            for cue_labels, ls, marker, fill in ((cues, "-", "o", color), (("no cue",), "--", "o", "none")):
                mean, std = [], []
                for dataset in datasets:
                    records = load_topdown(m, dataset, cue_type)
                    require_nonempty(m, f"top-down {dataset}/{cue_type}", len(records))
                    per_cue = [topdown_side_stats(records, cue_type, c, side_tokens)[1] for c in cue_labels]
                    mean.append(float(np.mean([p[0] for p in per_cue])))
                    std.append(float(np.mean([p[1] for p in per_cue])))
                mean, std = np.array(mean), np.array(std)
                if ls == "-":
                    ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                    alpha=0.12, color=color, linewidth=0)
                ax.plot(x, mean, marker=marker, markersize=2.5, linestyle=ls, color=color,
                        markerfacecolor=fill, label=f"{side_name} cue" if ls == "-" else "no cue")
        ax.set_xticks(x)
        ax.set_xticklabels(TD_EVIDENCE_LEVELS, fontsize=5.5)
        ax.set_xlim(-0.25, len(TD_EVIDENCE_LEVELS) - 0.75)
        tidy(ax, title=SHORT[m], ylim=(0, 1))
    for ax in axes[::ncols]:
        ax.set_ylabel("p(cued percept)")
    for r in range(math.ceil(len(models) / ncols)):
        center_xlabel(axes[r * ncols:(r + 1) * ncols], "Visual evidence for the cued percept")
    # colors distinguish the two cue directions, line style the cued/no-cue runs,
    # so the legend is built from styles rather than from the drawn handles (a
    # deduplicated "no cue" would otherwise take the colour of whichever cue drew
    # it first and read as belonging to that direction)
    handles = [Line2D([], [], color=c, marker="o", markersize=2.5, label=f"{n} cue")
               for _, n, _, c, _ in TD_EVIDENCE_CUES[cue_type]]
    handles.append(Line2D([], [], color="0.35", linestyle="--", marker="o", markersize=2.5,
                          markerfacecolor="none", label="no cue"))
    shared_legend(fig, handles, [h.get_label() for h in handles], anchor_axes=list(axes))
    savefig(fig, name)


def topdown_negation_row(name, models, cue_type, ncols=5, width=TEXTWIDTH):
    """Positive vs negated form of the same cue, as a paired shift from no cue.

    The claim is that a negated cue moves the report the *other* way,
    which the per-token panels could not show (they plot neither the balance nor
    the baseline). Here the two polarities sit on one axis in preference units,
    so a mirroring cue is a pair of points on opposite sides of zero.
    """
    models = available(models)
    fig, axes = grid(len(models), ncols, 1.25 if ncols >= 5 else 1.7, width=width,
                     hspace=1.0 if cue_type.startswith("prefix") else 1.5, wspace=0.3)
    for ax, m in zip(axes, models):
        ax.axhline(0.0, color="0.35", linewidth=0.6, zorder=1)
        panel_curves = []
        for suffix, label, ls, marker in (("", "cue", "-", "o"), ("_negation", "negated cue", "--", "s")):
            records = load_topdown(m, "va_all", cue_type + suffix)
            require_nonempty(m, f"top-down va_all/{cue_type}{suffix}", len(records))
            _, (counts, mean, std) = topdown_pref_stats(records, cue_type + suffix, relative=True)
            counts, mean, std = counts[1:], mean[1:], std[1:]  # no-cue column is identically zero
            panel_curves.append((mean, std))
            x = np.arange(len(mean))
            valid = counts >= 2
            if np.any(valid):
                ax.fill_between(x, mean - std, mean + std, where=valid, alpha=0.12,
                                color=TD_NEGATION_COLOR, linewidth=0)
            ax.plot(x, mean, marker=marker, markersize=2.5, linestyle=ls, color=TD_NEGATION_COLOR,
                    markerfacecolor="none" if suffix else TD_NEGATION_COLOR, label=label)
        ax.set_xticks(np.arange(len(TD_CUES[cue_type])))
        ax.set_xticklabels(TD_CUES[cue_type], rotation=40, ha="right", fontsize=5.5)
        tidy(ax, title=SHORT[m])
        autoscale_y(ax, panel_curves, floor=0.3, lo=-1.0, hi=1.0, symmetric=True)
    for ax in axes[::ncols]:
        ax.set_ylabel("Δ preference\n(vs no cue)")
    for r in range(math.ceil(len(models) / ncols)):
        center_xlabel(axes[r * ncols:(r + 1) * ncols], TD_XLABEL[cue_type])
    h, l = axes[0].get_legend_handles_labels()
    shared_legend(fig, h, l, anchor_axes=list(axes))
    savefig(fig, name)


def free_row(name, models, dataset, slug, ncols=5):
    models = available(models)
    fig, axes = grid(len(models), ncols, 1.2, hspace=0.75, wspace=0.4)
    for ax, m in zip(axes, models):
        rel, stats = load_free(m, dataset, slug)
        draw_free(ax, rel, stats)
        tidy(ax, xlabel="Angle" if dataset == "harper" else None, title=SHORT[m], ylim=(0, 1))
    for ax in axes[::ncols]:
        ax.set_ylabel("Probability")
    if dataset != "harper":
        for r in range(math.ceil(len(models) / ncols)):
            center_xlabel(axes[r * ncols:(r + 1) * ncols], "Angle (relative to boundary)")
    h, l = axes[0].get_legend_handles_labels()
    shared_legend(fig, h, l, anchor_axes=list(axes))
    savefig(fig, name)


def red_circle_row(name, models):
    models = available(models)
    n = len(models) + 1
    fig, axes = grid(n, n, 1.45, hspace=0.2, wspace=0.12)
    for ax, m in zip(axes, models):
        base, heat, vmax = red_circle_diff_map(m)
        im = draw_heat(ax, base, heat, vmax)
        ax.set_title(SHORT[m])
        cb = slim_colorbar(fig, im, ax)
        if m == models[len(models) // 2]:
            cb.set_label("Δ logit (rabbit − bird), baseline-subtracted", fontsize=6)
    draw_human(axes[-1], "Human")
    savefig(fig, name)


# ---------------- main text ----------------
@figure("main_rotation")
def _main_rotation():
    fig, axes = grid(3, 3, 1.0, hspace=0.6, wspace=0.75)
    for ax, ds, title in zip(axes, ["harper", "va_all", "split_all"],
                             ["(a) Harper's version", "(b) Visual Anagrams", "(c) Non-ambiguous controls"]):
        runs, centers = load_rotation(MAIN, ds)
        angles, curves, n = curve_stats(runs, centers, COLOR_DICT, max_curves=8)
        draw_curves(ax, angles, curves, n)
        tidy(ax, xlabel="Angle", title=title)
        right_legend(ax, maxlen=10)
    axes[0].set_ylabel("Probability")
    savefig(fig, "main_rotation")


@figure("main_red_circle")
def _main_red_circle():
    fig, axes = grid(2, 2, 1.25, width=COLWIDTH, wspace=0.25)
    base, heat, vmax = red_circle_diff_map(MAIN)
    im = draw_heat(axes[0], base, heat, vmax)
    axes[0].set_title("(a) LLaVA-1.5-7B")
    slim_colorbar(fig, im, axes[0], label="Δ logit (rabbit − bird)", location="right")
    draw_human(axes[1], "(b) Human")
    savefig(fig, "main_red_circle")


@figure("main_top_down")
def _main_top_down():
    fig, axes = grid(2, 1, 1.05, width=COLWIDTH, hspace=1.3)
    for ax, ct, title in zip(axes, ["prefix", "semantic"], ["(a) Prefix cues", "(b) Semantic cues"]):
        x_labels, stats = topdown_stats(load_topdown(MAIN, "va_all", ct), ct)
        draw_topdown(ax, x_labels, stats)
        tidy(ax, xlabel=TD_XLABEL[ct], ylabel="Probability")
        ax.set_title(title)
        right_legend(ax)
    savefig(fig, "main_top_down")


@figure("main_beam")
def _main_beam():
    fig, axes = grid(3, 3, 1.0, hspace=0.6, wspace=0.75)
    prob = load_beam_single(mp(BEAM_DIR, MAIN, "harper", f"{BASE_SLUG}-bird.json"), center=0)
    angles, curves, n = curve_stats([prob], [0], COLOR_DICT, beam_mode=True, max_curves=8)
    draw_curves(axes[0], angles, curves, n, label_fn=beam_label)
    tidy(axes[0], xlabel="Angle", ylabel="Probability", title="(a) Harper's version")
    panel_legend(axes[0], ncol=2, y=-0.36, maxlen=18)
    cl = load_classified()
    for ax, cls, title in zip(axes[1:], ["va", "split"], ["(b) Visual Anagrams", "(c) Non-ambiguous controls"]):
        draw_classified(ax, cl[f"{MAIN}|{BASE_SLUG}|{cls}"], cls)
        tidy(ax, xlabel="Angle", ylim=(0, 1), title=title)
    axes[1].set_ylabel("Share of beam mass")
    h, l = axes[1].get_legend_handles_labels()
    shared_legend(fig, h, l, anchor_axes=list(axes[1:]))
    savefig(fig, "main_beam")


@figure("main_dla")
def _main_dla():
    fig, ax = plt.subplots(figsize=(COLWIDTH, 2.2))
    draw_dla(ax, load_dla(MAIN), list(get_duck_rabbit_tokens(MAIN)))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Direct logit attribution")
    panel_legend(ax, ncol=2, y=-0.17, maxlen=30)   # sit right under the x label
    savefig(fig, "main_dla")


@figure("main_dpm_scatter")
def _main_dpm():
    fig, ax = plt.subplots(figsize=(COLWIDTH, 1.9))
    bird, rabb = get_duck_rabbit_tokens(MAIN)
    sc = draw_dpm(ax, load_dpm(MAIN))
    ax.set_xlabel(f"# dominant patches ({bird})")
    ax.set_ylabel(f"# dominant patches ({rabb})")
    ax.legend(loc="upper right")
    slim_colorbar(fig, sc, ax, label=f"Δ({bird} − {rabb})", location="right")
    savefig(fig, "main_dpm_scatter")


@figure("main_resampling")
def _main_resampling():
    fig, axes = grid(3, 3, 2.0, hspace=0.3, wspace=0.55)
    for ax, comp, letter in zip(axes, ["resid_post", "attn_out", "mlp_out"], "abc"):
        mat, labels = load_resampling(MAIN, comp)
        im = draw_resampling(ax, mat, labels, comp)
        ax.set_xlabel("Layer")
        ax.set_title(f"({letter}) {comp}")
        slim_colorbar(fig, im, ax, label="logit diff (duck − rabbit)", location="right")
    axes[0].set_ylabel("Token")
    savefig(fig, "main_resampling")


@figure("main_query_patching")
def _main_qp():
    fig, ax = plt.subplots(figsize=(COLWIDTH, 1.9))
    recs, fit = load_query_patching(MAIN)
    draw_query_patching(ax, recs, fit, annotate=False)
    ax.set_xlabel("Total modulation effect: Δ(duck − rabbit)")
    ax.set_ylabel("Effect explained by\nquery-side attention")   # shorter: the full label was clipped (Fig 11)
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)
    ax.text(0.03, 0.97, f"β(bottom-up) = {fit['bottom_up']['slope']:.3f}\nβ(top-down) = {fit['top_down']['slope']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6)
    savefig(fig, "main_query_patching")


@figure("main_exclusivity")
def _main_excl():
    fig = plt.figure(figsize=(COLWIDTH, 2.6))
    fig.subplots_adjust(left=0.17, right=0.99, top=0.99, bottom=0.14)
    ax, _, _ = draw_exclusivity_joint(
        fig, load_exclusivity(MAIN),
        xlabel='Count effect: Δ(logit("2") − logit("1"))',
        ylabel='Punct effect: Δ(logit(" and") − logit("."))')
    ax.legend(handles=EXCL_LEGEND, loc="upper left", ncol=1, fontsize=5)
    savefig(fig, "main_exclusivity")


@figure("head_attn_maps")
def _head_attn_maps():
    """Fig 7: per-head image-attention contribution maps (top-4 / bottom-4).

    Rendered from the plot-input cache written by
    head_attention_maps.py (…_heads.npz: every (layer, head)
    map plus its summed contribution), so the plot can be restyled on CPU.
    """
    d = mp(os.path.join(OUTPUTS_DIR, "head_attention_maps", "plots", "head_dla"), MAIN,
           "harper", BASE_SLUG, "exact_attn_by_dla", "image_post_softmax")
    bird, rabb = get_duck_rabbit_tokens(MAIN)
    z = np.load(os.path.join(d, f"imgattn_{bird} - {rabb}_heads.npz"), allow_pickle=False)
    order = np.argsort(-z["sums"])
    top, bottom = order[:4], order[::-1][:4]
    vmax = float(max(np.max(np.abs(z["maps"][i])) for i in list(top) + list(bottom))) + 1e-9
    base, grid = z["image"], z["maps"].shape[-1]
    up = base.shape[0] // grid

    fig, axes = plt.subplots(2, 4, figsize=(COLWIDTH, 1.75))
    fig.subplots_adjust(left=0.01, right=0.86, top=0.90, bottom=0.02, wspace=0.06, hspace=0.28)
    im = None
    for row, idxs, tag in ((0, top, "top"), (1, bottom, "bottom")):
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(base, cmap="gray")
            im = ax.imshow(np.kron(z["maps"][i], np.ones((up, up))), cmap="RdBu",
                           vmin=-vmax, vmax=vmax, alpha=0.8)
            ax.set_title(f"L{z['layers'][i]} H{z['heads'][i]}", fontsize=5.5, pad=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel("bird-side" if tag == "top" else "rabbit-side", fontsize=5.5)
    cax = fig.add_axes([0.875, 0.06, 0.02, 0.84])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("per-head image contribution", fontsize=5.5)
    savefig(fig, "head_attn_maps")


@figure("dominant_patch_map")
def _dominant_patch_map():
    """Fig 8: dominant-patch map from the image-token logit lens.

    Derived on CPU from the cached logit-lens probabilities written by
    layerwise_dla.py; the derived map is cached next to them.
    """
    bird, rabb = get_duck_rabbit_tokens(MAIN)
    cdir = mp(os.path.join(DLA_DIR, "cache"), MAIN, "harper", BASE_SLUG)
    probs = np.load(os.path.join(cdir, "resid_post_logitlens_softmax.npy"))
    dpm = dominant_patch_map_from_logitlens(probs, tokens_of_interest={bird: 0, rabb: 1}, threshold=0.1)
    np.savez_compressed(os.path.join(cdir, "dominant_patch_map_bird_rabb.npz"),
                        dpm=dpm.astype(np.float32), tokens=np.array([bird, rabb]))
    grid = int(round(math.sqrt(dpm.shape[0])))
    diff = (dpm[..., 0] - dpm[..., 1]).reshape(grid, grid)
    base = np.asarray(RotatedRGBAImageDataset(os.path.join(STIM_DIR, "harper.png"), [0])[0].convert("L"))
    up = base.shape[0] // grid

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.imshow(base, cmap="gray")
    im = ax.imshow(np.kron(diff, np.ones((up, up))), cmap="RdBu", vmin=-1, vmax=1, alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("max prob diff: bird − rabbit", fontsize=5.5)
    savefig(fig, "dominant_patch_map")


# ---------------- appendix: behavioral ----------------
@figure("rotation_harper_llava")
def _(): rotation_row("rotation_harper_llava", LLAVA5, "harper")
@figure("rotation_va_all_llava")
def _(): rotation_row("rotation_va_all_llava", LLAVA5, "va_all")
@figure("rotation_harper_nonllava")
def _(): rotation_row("rotation_harper_nonllava", NONLLAVA4, "harper")
@figure("rotation_va_all_nonllava")
def _(): rotation_row("rotation_va_all_nonllava", NONLLAVA4, "va_all")

@figure("beam_bird_llava")
def _(): beam_row("beam_bird_llava", LLAVA5, "bird")
@figure("beam_rabbit_llava")
def _(): beam_row("beam_rabbit_llava", LLAVA5, "rabbit")

@figure("raven_classified_llava")
def _(): classified_row("raven_classified_llava", LLAVA5, BASE_SLUG, "raven_bear")
# --- forced-choice prompts -----------------------------------------------------
# "Is it a bird or a rabbit?" and the reversed order, prefix "It is a", first
# token over -45..45 at 1 deg RELATIVE to each va image's default-query boundary
# (rotation_first_token --center-on-boundary), so the panels share the -45..45 window and the
# token-level drawing of the rotation figures (Fig. 2 / rotation rows).
FC_DIR = os.path.join(OUTPUTS_DIR, "forced_choice", "text")
FC_ORDERS = (('"bird or rabbit?"', slugify("Is it a bird or a rabbit?" + "It is a"), "-"),
             ('"rabbit or bird?"', slugify("Is it a rabbit or a bird?" + "It is a"), "--"))


def load_forced_choice(model, order_slug):
    """Like load_rotation(va_all): per-stimulus top-k runs + boundary centres."""
    runs, centers = [], []
    for va in valid_va_names(model):
        path = mp(FC_DIR, model, va, f"{order_slug}.json")
        if not os.path.exists(path):
            continue
        center = get_boundary(BOUNDARIES, model, va, BASE_SLUG)
        runs.append(extract_topk_from_file(path, k=3, extra_tokens=list(COLOR_DICT), anchor_angle=center))
        centers.append(center)
    require_nonempty(model, "forced-choice", len(runs))
    return runs, centers


# The question names "bird" and "rabbit", so only those two tokens are tracked
# (no duck / bunny curves).
FC_TOKENS = tuple(t for t, c in COLOR_DICT.items() if c in ("skyblue", "orange"))


def forced_choice_row(name, models, ncols=5, width=TEXTWIDTH):
    """Bird / rabbit tokens per model; solid = bird named first, dashed =
    rabbit named first. SD bands on the solid order only."""
    models = available(models)
    fig, axes = grid(len(models), ncols, 1.25, width=width, hspace=1.1, wspace=0.4)
    for ax, m in zip(axes, models):
        for _label, slug, ls in FC_ORDERS:
            runs, centers = load_forced_choice(m, slug)
            angles, curves, n = curve_stats(runs, centers, COLOR_DICT)
            curves = [(tok, c, mu, sd, col, ls) for tok, c, mu, sd, col, _ in curves if tok in FC_TOKENS]
            draw_curves(ax, angles, curves, n if ls == "-" else 1,
                        label_fn=(lambda t: t) if ls == "-" else (lambda t: "_nolegend_"))
        tidy(ax, xlabel="Angle", title=SHORT[m])
        panel_legend(ax, y=-0.42)
    for ax in axes[::ncols]:
        ax.set_ylabel("Probability")
    handles = [Line2D([], [], color="0.35", lw=1.2, ls=ls) for _, _, ls in FC_ORDERS]
    shared_legend(fig, handles, [lab for lab, _, _ in FC_ORDERS], anchor_axes=list(axes))
    savefig(fig, name)


@figure("forced_choice_nine")
def _(): forced_choice_row("forced_choice_nine", NINE)


@figure("classified_va_base_llava")
def _(): classified_row("classified_va_base_llava", LLAVA5, BASE_SLUG, "va")
@figure("classified_va_base_nonllava")
def _(): classified_row("classified_va_base_nonllava", NONLLAVA4, BASE_SLUG, "va")
@figure("classified_split_base_llava")
def _(): classified_row("classified_split_base_llava", LLAVA5, BASE_SLUG, "split")
@figure("classified_va_hint_nine")
def _(): classified_row("classified_va_hint_nine", NINE, HINT_SLUG, "va")

@figure("red_circle_llava")
def _(): red_circle_row("red_circle_llava", LLAVA5)
@figure("red_circle_nonllava")
def _(): red_circle_row("red_circle_nonllava", NONLLAVA4)

# Positive cues on the ambiguous stimuli, per token (B.1 / B.2).
for _ct in ("prefix", "semantic"):
    for _grp, _models in (("llava", LLAVA5), ("nonllava", NONLLAVA4)):
        _name = f"topdown_{_ct}_va_all_{_grp}"
        FIGS[_name] = (lambda n=_name, ms=_models, c=_ct: topdown_row(n, ms, "va_all", c))
# Negative cues (B.6): positive and negated form of each cue as a paired shift
# from no cue, so "the negation moves the report the other way" is readable.
for _ct in ("prefix", "semantic"):
    _name = f"topdown_negation_{_ct}_nine"
    FIGS[_name] = (lambda n=_name, c=_ct: topdown_negation_row(n, NINE, c, ncols=5))
# Boundary analyses (B.6): ambiguous vs single-object stimuli on one axis, in
# preference units, so "the 'd' cue does not work on a rabbit-only image" is
# the readable claim.
for _ct in ("prefix", "semantic"):
    _name = f"topdown_evidence_{_ct}_nine"
    FIGS[_name] = (lambda n=_name, c=_ct: topdown_evidence_row(n, NINE, c, ncols=5))

for _ds in ("va_all",):
    for _key, _slug in (("general", "describe-this-image"), ("cot", "describe-this-image-think-step-by-step")):
        _name = f"free_{_key}_{_ds}_nine"
        FIGS[_name] = (lambda n=_name, d=_ds, s=_slug: free_row(n, NINE, d, s))


# ---------------- appendix: figure-ground ----------------
@figure("raven_topdown_llava")
def _raven_topdown():
    cues = ["b", "d", "x"]
    fig, axes = grid(5, 5, 1.25, hspace=0.6, wspace=0.4)
    for ax, m in zip(axes, LLAVA5):
        tok_a, tok_b = get_raven_bear_tokens(m)
        rec = {}
        red = demangle_path(mp(RB_DIR, m, "red_circle", f"{BASE_SLUG}_{tok_a}_{tok_b}_r0p071_s0p143_lw4.json"))
        with open(red) as f:
            for it in json.load(f):
                if it.get("center", "missing") is None and it.get("output"):
                    b = it["output"][0]
                    rec["no cue"] = {"prob_data": [{"angle": 0, "tokens": b["tokens"], "probs": b["probs"]}], "center_angle": 0}
                    break
        for cue in cues:
            p = mp(RB_DIR, m, "top_down", f"{slugify(TD_BASE + f' It starts with {cue!r}.' + 'I see a')}.json")
            if os.path.exists(p):
                rec[cue] = {"prob_data": extract_topk_from_file(p, k=5, extra_tokens=[tok_a, tok_b]), "center_angle": 0}
        color_dict = {tok_a: "skyblue", tok_b: "orange"}
        x_labels = ["no cue"] + cues
        x = np.arange(len(x_labels))
        for tok, col in color_dict.items():
            y = [dict(zip(r["prob_data"][0]["tokens"], r["prob_data"][0]["probs"])).get(tok, np.nan) if (r := rec.get(c)) else np.nan
                 for c in x_labels]
            ax.plot(x, y, marker="o", markersize=2.5, color=col, label=tok)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=40, ha="right")
        tidy(ax, title=SHORT[m], ylim=(0, 1))
        panel_legend(ax, ncol=2, y=-0.5)
    axes[0].set_ylabel("Probability")
    row_xlabel(fig, axes, "It starts with '\u2026'.", dy=-0.2)
    savefig(fig, "raven_topdown_llava")


@figure("va_gallery")
def _va_gallery():
    """Appendix A stimulus gallery: 10 VA stimuli, duck view (+45 deg) over
    rabbit view (-45 deg), in experiment geometry, each tile outlined."""
    chosen = GALLERY_STIMULI
    fig, axes = plt.subplots(2, 10, figsize=(TEXTWIDTH, 1.45))
    fig.subplots_adjust(left=0.035, right=0.999, top=0.98, bottom=0.02, wspace=0.06, hspace=0.06)
    for col, name in enumerate(chosen):
        path = os.path.join(STIM_DIR, f"{name}.png")
        for row, angle in ((0, 45), (1, -45)):
            ax = axes[row, col]
            ax.imshow(RotatedRGBAImageDataset(path, [angle])[0])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():          # the per-image outline
                sp.set_linewidth(0.5)
                sp.set_color("0.35")
    axes[0, 0].set_ylabel("duck view\n($+45^\\circ$)", fontsize=6)
    axes[1, 0].set_ylabel("rabbit view\n($-45^\\circ$)", fontsize=6)
    savefig(fig, "va_gallery")


# ---------------- appendix: mechanistic ----------------
@figure("dla_llava")
def _dla_row():
    fig, axes = grid(5, 5, 1.3, hspace=0.6, wspace=0.4)
    for ax, m in zip(axes, LLAVA5):
        draw_dla(ax, load_dla(m), ["bird", "rabbit"])
        ax.set_xlabel("Layer")
        ax.set_title(SHORT[m])
    axes[0].set_ylabel("Direct logit attribution")
    h, l = axes[0].get_legend_handles_labels()
    shared_legend(fig, h, l, ncol=4, anchor_axes=list(axes))
    savefig(fig, "dla_llava")


@figure("dpm_scatter_llava")
def _dpm_row():
    fig, axes = grid(5, 5, 1.35, hspace=0.6, wspace=0.45)
    recs = {m: load_dpm(m) for m in LLAVA5}
    vmax = max(float(np.nanmax(np.abs([r["score_diff"] for r in rs if np.isfinite(r["score_diff"])]))) for rs in recs.values())
    for ax, m in zip(axes, LLAVA5):
        sc = draw_dpm(ax, recs[m], vmax=vmax)
        ax.set_title(SHORT[m])
    center_xlabel(axes, "# bird-dominant patches")
    axes[0].set_ylabel("# rabbit-dominant patches")
    cb = fig.colorbar(sc, ax=list(axes), location="right", fraction=0.02, pad=0.01, shrink=0.8)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Δ logit (bird − rabbit)", fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    shared_legend(fig, h, l, anchor_axes=list(axes))
    savefig(fig, "dpm_scatter_llava")


def resampling_row(name, component):
    fig, axes = grid(5, 5, 2.4, hspace=0.3, wspace=0.75)
    for ax, m in zip(axes, LLAVA5):
        mat, labels = load_resampling(m, component)
        im = draw_resampling(ax, mat, labels, component)
        ax.set_xlabel("Layer")
        ax.set_title(SHORT[m])
        slim_colorbar(fig, im, ax, location="right", label="logit diff (duck − rabbit)" if m == LLAVA5[-1] else None)
    savefig(fig, name)


for _comp in ("resid_post", "attn_out", "mlp_out"):
    FIGS[f"resampling_{_comp}_llava"] = (lambda n=f"resampling_{_comp}_llava", c=_comp: resampling_row(n, c))


@figure("query_patching_llava3")
def _qp_row():
    models = ["llava-1.5-7b", "llava-1.5-13b", "llava-v1.6-vicuna-7b"]
    fig, axes = grid(3, 3, 1.9, hspace=0.5, wspace=0.35)
    for ax, m in zip(axes, models):
        recs, fit = load_query_patching(m)
        draw_query_patching(ax, recs, fit)
        ax.set_title(SHORT[m])
    center_xlabel(axes, "Total modulation effect: Δ(duck − rabbit)")
    axes[0].set_ylabel("Effect explained by query-side\ntext→image attention")
    h, l = axes[0].get_legend_handles_labels()
    shared_legend(fig, h, l, anchor_axes=list(axes))
    savefig(fig, "query_patching_llava3")


@figure("exclusivity_llava")
def _excl_row():
    models = available(LLAVA5)
    fig = plt.figure(figsize=(TEXTWIDTH, 1.9))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.22)
    outer = fig.add_gridspec(1, len(models), wspace=0.5)
    mains = []
    for i, m in enumerate(models):
        ax, ax_top, _ = draw_exclusivity_joint(fig, load_exclusivity(m), gs=outer[0, i],
                                               hist_frac=0.18, label_counts=False)
        ax_top.set_title(SHORT[m], fontsize=7.5, pad=2)
        ax_top.set_ylabel("")
        mains.append(ax)
    center_xlabel(mains, 'Count effect: Δ("2" − "1")')
    mains[0].set_ylabel('Punct effect: Δ(" and" − ".")')
    shared_legend(fig, EXCL_LEGEND, [h.get_label() for h in EXCL_LEGEND], anchor_axes=mains)
    savefig(fig, "exclusivity_llava")


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="substring filter on figure names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use-quarantined", action="store_true",
                    help="fall back to <model>__* dirs when <model> is absent (reports every use)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="drop models whose result dirs are absent instead of failing (prints them at the end)")
    args = ap.parse_args()
    ALLOW_MISSING = args.allow_missing
    USE_QUARANTINED = args.use_quarantined
    names = [n for n in FIGS if (args.only is None or args.only in n)]
    if args.list:
        print("\n".join(names))
        sys.exit(0)
    failed = []
    for n in names:
        try:
            FIGS[n]()
        except Exception as e:  # keep going: one missing cache must not block the wave
            import traceback
            traceback.print_exc()
            failed.append((n, repr(e)))
            plt.close("all")
    if QUARANTINED_USED:
        print(f"\n[WARN] read from fallback result dirs: {sorted(QUARANTINED_USED)}")
    if MISSING_MODELS:
        print(f"\n[WARN] models absent from outputs/ and dropped from every row: {sorted(MISSING_MODELS)}")
    print(f"\n{len(names) - len(failed)}/{len(names)} figures written to {OUT_DIR}")
    for n, e in failed:
        print(f"[FAILED] {n}: {e}")
    sys.exit(1 if failed else 0)
