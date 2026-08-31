# Keyword-v2 classifier for the free-form figures.
#   vocabulary : the judge's noun table (bird / rabbit / other), which was built
#                unprimed from the models' own outputs
#   matching   : WORD BOUNDARIES with optional plural (never substrings, so
#                "ant" cannot fire inside "pants"); irregular plurals listed
#   classes    : duck / rabbit / both / other / none   (none = no animal noun)
#   validation : agreement against the LLM-judge labels on the judged subset
# Runs on the FULL data (all samples, all angles) at zero API cost.
import argparse, glob, hashlib, json, os, re, sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from importlib import import_module
from utils import PROJECT_ROOT
_v3 = import_module("analysis.llm_judge")
PROMPT_VERSION = _v3.PROMPT_VERSION
_ff = import_module("analysis.judge_freeform_curves")
label_from_nouns, label_percept_wins = _ff.label_from_nouns, _ff.label_percept_wins
CLASSES = ("single", "duck", "rabbit", "multiple", "both", "none")
DISPLAY = {"single": "single animal", "duck": "bird only", "rabbit": "rabbit only",
           "multiple": "multiple animals", "both": "both percepts", "none": "no animal"}
COLORS = {"single": "#1b7f3b", "duck": "tab:blue", "rabbit": "tab:orange",
          "multiple": "#8e44ad", "both": "#c71585", "none": "gray"}

IRREGULAR = {"goose": "geese", "bunny": "bunnies", "butterfly": "butterflies", "mouse": "mice",
             "ox": "oxen", "man": "men", "woman": "women", "person": "people", "calf": "calves",
             "wolf": "wolves", "sheep": "sheep", "deer": "deer", "fish": "fish"}


# Generic terms are not species evidence: they appear in ~18% of descriptions
# ("the animal's ear"), and under the v4 rule a single other-class hit overrides
# the percept. The judge lists them only when the text really names them as the
# subject, so keyword matching must skip them.
# Body parts are not animals, but step 2 of the judge ("is a beak bird-like?")
# honestly answers yes, so they entered the noun table and fired often ("beak"
# 202x). Excluded explicitly.
BODY_PARTS = {"beak", "beaks", "bill", "bills", "ear", "ears", "eye", "eyes", "fur", "feather",
              "feathers", "wing", "wings", "tail", "tails", "paw", "paws", "snout", "snouts",
              "muzzle", "whisker", "whiskers", "hoof", "hooves", "horn", "horns", "antler",
              "antlers", "claw", "claws", "nose", "noses", "head", "heads", "body", "face",
              "faces", "mouth", "mouths", "leg", "legs", "foot", "feet", "fin", "fins", "prey"}

GENERIC = {"animal", "animals", "creature", "creatures", "beast", "beasts", "mammal", "mammals",
           "rodent", "rodents", "insect", "insects", "reptile", "reptiles", "feline", "species",
           "human", "humans", "person", "people", "man", "men", "woman", "women", "figure"}


def build_matcher(noun_table, listing_counts=None, min_listings=20):
    """ONE combined word-boundary regex over every surface form (300 separate
    regexes over ~2.4M texts is far too slow); surface form -> canonical noun."""
    form2noun, cls = {}, {}
    for noun, c in noun_table.items():
        if c not in ("bird", "rabbit", "other") or not noun.isalpha():
            continue
        if noun in GENERIC or noun in BODY_PARTS:
            continue
        # A handful of judge answers were sentences, not lists, so the noun table
        # picked up junk ("the", "is", "ears"); those words were listed 1-14 times
        # while real animals appear thousands of times. Require evidence.
        if listing_counts is not None and listing_counts.get(noun, 0) < min_listings:
            continue
        forms = {noun, noun + "s", noun + "es"}
        if noun in IRREGULAR:
            forms.add(IRREGULAR[noun])
        if noun.endswith("y"):
            forms.add(noun[:-1] + "ies")
        for f in forms:
            form2noun.setdefault(f.lower(), noun)
        cls[noun] = c
    big = re.compile(r"\b(?:%s)\b" % "|".join(sorted(form2noun, key=len, reverse=True)), re.I)
    return (big, form2noun), cls


def nouns_in(text, pats):
    big, form2noun = pats
    return {form2noun[m.group(0).lower()] for m in big.finditer(text.lower())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle-lim", type=int, default=60, help="full plotted range by default")
    ap.add_argument("--judge-model", default="gpt-5-nano-2025-08-07")
    ap.add_argument("--rule", default="v4", choices=["v4", "percept_wins"])
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Use only the first N samples per VA image and angle (harper keeps all). "
                         "2 = the Llama3 rerun's budget, applied to every model for comparability.")
    args = ap.parse_args()

    root = PROJECT_ROOT
    man_dir = os.path.join(root, "outputs", "llm_judge", "manifests",
                           f"{args.judge_model}__{PROMPT_VERSION}")
    noun_table = json.load(open(os.path.join(man_dir, "noun_table.json")))
    for f in glob.glob(os.path.join(man_dir, "noun_table_add__*.json")):   # merge per-job additions
        noun_table.update(json.load(open(f)))
    listing_counts = Counter()
    for f in glob.glob(os.path.join(man_dir, "freeform__*.json")):
        for v in json.load(open(f)).values():
            for n in v.get("nouns", []):
                listing_counts[n] += 1
    pats, cls_of = build_matcher(noun_table, listing_counts)
    print(f"keyword-v2 vocabulary: {len(cls_of)} nouns "
          f"({sum(1 for n in cls_of if cls_of[n]=='bird')} bird / "
          f"{sum(1 for n in cls_of if cls_of[n]=='rabbit')} rabbit / "
          f"{sum(1 for n in cls_of if cls_of[n]=='other')} other)", flush=True)

    # Classes mirror the beam aggregate panels: the partition
    # is single / multiple / none by the NUMBER of distinct animals named, with
    # duck-only and rabbit-only as subsets of single and both-percepts as a subset
    # of multiple. "a rabbit and a cat" is therefore multiple animals -- not
    # "rabbit" (which would hide the second animal) and not "other" (which would
    # hide the reported percept).
    def label_counts(nouns):
        # Count ANIMALS, not nouns: "a bunny ... the rabbit" names one animal, and
        # "a duck ... the bird" likewise. All bird-class nouns collapse to one
        # entity, all rabbit-class nouns to one, and each distinct other-animal
        # noun counts separately (this is what the judge does when it lists the
        # animals it thinks are named).
        nn = set(nouns)
        cls = {noun_table.get(n) for n in nn}
        cls = {c for c in cls if c in ("bird", "rabbit", "other")}
        k = (1 if "bird" in cls else 0) + (1 if "rabbit" in cls else 0) \
            + len({n for n in nn if noun_table.get(n) == "other"})
        out = Counter()
        if k == 0:
            out["none"] = 1
            return out
        if k == 1:
            out["single"] = 1
            if "bird" in cls:
                out["duck"] = 1
            elif "rabbit" in cls:
                out["rabbit"] = 1
            return out
        out["multiple"] = 1
        if "bird" in cls and "rabbit" in cls:
            out["both"] = 1
        return out

    label = label_counts
    out_root = os.path.join(root, "outputs", "keyword_freeform")
    os.makedirs(os.path.join(out_root, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "plots"), exist_ok=True)

    # judge labels for validation (all judged texts, any model/prompt)
    judged_nouns = {}
    for f in glob.glob(os.path.join(man_dir, "freeform__*.json")):
        for k, v in json.load(open(f)).items():
            if v.get("raw", "").strip():
                judged_nouns[k] = v["nouns"]
    print(f"judge labels available for validation: {len(judged_nouns)}", flush=True)

    models = ["llava-1.5-7b", "llava-1.5-13b", "llava-v1.6-vicuna-7b", "llava-v1.6-mistral-7b",
              "llama3-llava-next-8b", "Qwen2-VL-7B-Instruct", "smolvlm-2b", "idefics2-8b",
              "instructblip-vicuna-7b"]
    slugs = ["describe-this-image", "describe-this-image-think-step-by-step"]
    val = Counter()
    summary = {}
    for model in models:
        src = os.path.join(root, "outputs", "rotation_freeform", "text", model)
        if not os.path.isdir(src):
            print(f"[SKIP] {model}: no free-form outputs", flush=True)
            continue
        for slug in slugs:
            curves = defaultdict(lambda: defaultdict(Counter))
            per_image = {}          # image -> [{angle, duck, rabbit, both, other, none}]
            n_img = 0
            for img_dir in sorted(glob.glob(os.path.join(src, "*"))):
                image = os.path.basename(img_dir)
                f = os.path.join(img_dir, f"{slug}.json")
                if not os.path.exists(f):
                    continue
                if image == "harper":
                    cl = "harper"
                elif image.startswith("va_s"):
                    cl = "va_all"; n_img += 1
                else:
                    continue
                data = json.load(open(f))
                angs = [it["angle"] for it in data]
                center = (min(angs) + max(angs)) / 2.0
                rows_img = []
                for it in data:
                    rel = int(round(it["angle"] - center))
                    cnt_ang = Counter()
                    outs = it["output"]
                    if args.max_samples is not None and cl == "va_all":
                        outs = outs[:args.max_samples]
                    for t in outs:
                        flags = label(nouns_in(t, pats))
                        cnt_ang["_n"] += 1
                        for k_ in flags:
                            cnt_ang[k_] += 1
                            if abs(rel) <= args.angle_lim:
                                curves[cl][rel][k_] += 1
                        if abs(rel) <= args.angle_lim:
                            curves[cl][rel]["_n"] += 1
                        # validation against the judge uses the same scheme
                        e_j = judged_nouns.get(hashlib.sha1(t.strip().encode()).hexdigest())
                        if e_j is not None:
                            fj = label(e_j)
                            val[("match" if set(fj) == set(flags) else "differ",
                                 "/".join(sorted(fj)), "/".join(sorted(flags)))] += 1
                    ntot = cnt_ang.get("_n", 0)
                    if ntot:
                        rows_img.append({"angle": it["angle"],
                                         **{k: cnt_ang.get(k, 0) / ntot for k in CLASSES}})
                per_image[image] = rows_img
            if not curves:
                continue
            # per-image probs in the rotation_freeform layout: probs/<model>/<image>/<slug>.json
            for image, rows in per_image.items():
                d = os.path.join(out_root, "probs", model, image)
                os.makedirs(d, exist_ok=True)
                json.dump(rows, open(os.path.join(d, f"{slug}.json"), "w"), indent=1)
            tag = f"{model}__{slug}"
            summary[tag] = {cl: [dict(rel_angle=r, n=c.get("_n", 0),
                                      **{k: c.get(k, 0) / max(c.get("_n", 1), 1) for k in CLASSES})
                                 for r, c in sorted(byrel.items())] for cl, byrel in curves.items()}
            for cl, rows in summary[tag].items():
                x = [r["rel_angle"] for r in rows]
                fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
                for lab in CLASSES:
                    ax.plot(x, [r[lab] for r in rows], label=DISPLAY[lab], color=COLORS[lab])
                ax.set_xlabel("Angle (relative to center)"); ax.set_ylabel("Probability")
                ax.set_ylim(0, 1); ax.set_xlim(min(x), max(x)); ax.grid(True); ax.legend()
                fig.savefig(os.path.join(out_root, "plots", f"{model}__{slug}__{cl}.png"),
                            bbox_inches="tight", pad_inches=0.05)
                plt.close(fig)
            print(f"{tag}: {n_img} VA images, angles ±{args.angle_lim}, n/angle(va_all)="
                  f"{summary[tag].get('va_all', [{'n': 0}])[0]['n']}", flush=True)

    m = sum(v for (s, _, _), v in val.items() if s == "match"); n = sum(val.values())
    print(f"\nkeyword-v2 vs LLM judge: {m/max(n,1):.4f} agreement over {n} texts", flush=True)
    conf = Counter()
    for (s, j, k), v in val.items():
        if s == "differ":
            conf[f"judge={j} -> kwv2={k}"] += v
    for k, v in conf.most_common(6):
        print(f"   {k:28s} {v:6d} ({100*v/max(n,1):.2f}%)", flush=True)
    json.dump({"rule": args.rule, "angle_lim": args.angle_lim, "vocabulary": len(pats),
               "agreement_vs_judge": m / max(n, 1), "n_validated": n,
               "disagreements": dict(conf), "curves": summary},
              open(os.path.join(out_root, "tables", f"keyword_v2_{args.rule}.json"), "w"), indent=1)
    print("done", flush=True)
