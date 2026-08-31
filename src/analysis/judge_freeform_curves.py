# LLM-judge classification of the FREE-FORM generations
# (rotation_freeform: "Describe this image." and the CoT variant), replacing keyword
# matching in the B.5 figures. Protocol:
#   angles   : relative -45..+45 (what the figures plot)
#   samples  : first 2 of 8 per va_s* image x angle; first 32 of 64 for harper
#   classes  : duck-only / rabbit-only / both / other / none, pooled across all
#              texts at each angle (each sample is a categorical draw, so the
#              per-angle quantity is a proportion -> no SD band; n in caption)
#   judge    : the v3 two-step pipeline, prompts imported verbatim
#              (list animals -> per-noun bird/rabbit/other table -> aggregate)
# Keyword labels are computed on the same texts only to report agreement.
import argparse, glob, hashlib, json, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from importlib import import_module
from utils import PROJECT_ROOT
_v3 = import_module("analysis.llm_judge")
LIST_PROMPT, NOUN_PROMPT, PROMPT_VERSION = _v3.LIST_PROMPT, _v3.NOUN_PROMPT, _v3.PROMPT_VERSION
parse_nouns, ask_once = _v3.parse_nouns, _v3.ask_once
keyword_label = _v3.classify_sentence_keyword_matching  # duck/rabbit/both/other (substring rule)

CLASSES = ("duck", "rabbit", "both", "other", "none")
DISPLAY = {"duck": "duck-only", "rabbit": "rabbit-only", "both": "both", "other": "other", "none": "none"}
COLORS = {"duck": "C0", "rabbit": "C1", "both": "C2", "other": "C3", "none": "gray"}


def label_percept_wins(nouns, noun_table):
    """Alternative rule: a co-mentioned other animal does not erase the percept
    ("a rabbit and a cat" -> rabbit). Matches the keyword classifier's logic,
    which only looks for duck/rabbit words; kept alongside the v4 rule so the
    figures can use either (both come free from the same cached judgments)."""
    classes = [noun_table.get(n) for n in set(nouns)]
    s = {c for c in classes if c in ("bird", "rabbit", "other")}
    if not s:
        return "none"
    if "bird" in s and "rabbit" in s:
        return "both"
    if "bird" in s:
        return "duck"
    if "rabbit" in s:
        return "rabbit"
    return "other"


def label_from_nouns(nouns, noun_table):
    classes = [noun_table.get(n) for n in set(nouns)]
    classes = [c for c in classes if c in ("bird", "rabbit", "other")]
    s = set(classes)
    if not s:
        return "none"
    if "bird" in s and "rabbit" in s:
        return "both"
    if "other" in s:
        return "other"          # any other animal present (v4 rule)
    return "duck" if s == {"bird"} else "rabbit"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-slug", default="describe-this-image")
    ap.add_argument("--judge-model", default="gpt-5-nano-2025-08-07")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--angle-lim", type=int, default=45)
    ap.add_argument("--va-samples", type=int, default=2)
    ap.add_argument("--harper-samples", type=int, default=32)
    ap.add_argument("--count-only", action="store_true")
    args = ap.parse_args()

    root = PROJECT_ROOT
    src_dir = os.path.join(root, "outputs", "rotation_freeform", "text", args.model)
    out_root = os.path.join(root, "outputs", "judge_freeform")
    man_dir = os.path.join(root, "outputs", "llm_judge", "manifests",
                           f"{args.judge_model}__{PROMPT_VERSION}")
    for d in (out_root, man_dir, os.path.join(out_root, "tables"), os.path.join(out_root, "plots")):
        os.makedirs(d, exist_ok=True)
    tag = f"{args.model}__{args.prompt_slug}"
    man_path = os.path.join(man_dir, f"freeform__{tag}.json")
    noun_table_path = os.path.join(man_dir, "noun_table.json")

    # ---- collect the subset ----
    # (image class, rel_angle) -> list of texts
    subset = defaultdict(list)
    n_va_images = 0
    for img_dir in sorted(glob.glob(os.path.join(src_dir, "*"))):
        image = os.path.basename(img_dir)
        f = os.path.join(img_dir, f"{args.prompt_slug}.json")
        if not os.path.exists(f):
            continue
        if image == "harper":
            cls, keep = "harper", args.harper_samples
        elif image.startswith("va_s"):
            cls, keep = "va_all", args.va_samples
            n_va_images += 1
        else:
            continue  # split controls are not plotted in the free-form figures
        data = json.load(open(f))
        angs = [it["angle"] for it in data]
        center = (min(angs) + max(angs)) / 2.0  # rel window is symmetric by construction
        for it in data:
            rel = int(round(it["angle"] - center))
            if abs(rel) <= args.angle_lim:
                subset[(cls, rel)].extend(it["output"][:keep])

    all_texts = [t.strip() for v in subset.values() for t in v]
    uniq = {hashlib.sha1(t.encode()).hexdigest(): t for t in all_texts}
    print(f"{tag}: {n_va_images} VA images, {len(all_texts)} texts in subset, {len(uniq)} unique", flush=True)
    if args.count_only:
        sys.exit(0)

    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    noun_table = json.load(open(noun_table_path)) if os.path.exists(noun_table_path) else {}
    todo = [(k, t) for k, t in uniq.items() if k not in manifest]
    print(f"  cached {len(manifest)}, to judge {len(todo)}", flush=True)

    from openai import OpenAI
    client = OpenAI()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ask_once, client, args.judge_model, LIST_PROMPT.format(text=t)): k for k, t in todo}
        for i, (fut, k) in enumerate(futs.items()):
            raw = fut.result()
            manifest[k] = {"raw": raw, "nouns": parse_nouns(raw)}
            if (i + 1) % 500 == 0:
                json.dump(manifest, open(man_path, "w"), ensure_ascii=False)
                print(f"  judged {i+1}/{len(todo)}", flush=True)
    json.dump(manifest, open(man_path, "w"), ensure_ascii=False)

    new_nouns = sorted({n for v in manifest.values() for n in v["nouns"]} - set(noun_table))
    if new_nouns:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(ask_once, client, args.judge_model, NOUN_PROMPT.format(noun=n)): n for n in new_nouns}
            for fut, n in futs.items():
                raw = fut.result().strip().lower()
                c = raw.split()[0].strip(".,\"'") if raw else "unparsed"
                noun_table[n] = c if c in ("bird", "rabbit", "other") else "unparsed"
        # 18 jobs run concurrently: write our additions to a per-job file and
        # merge into the shared table afterwards (last-writer-wins would drop
        # other jobs' nouns). Labels below use our own complete in-memory copy.
        json.dump({n: noun_table[n] for n in new_nouns},
                  open(os.path.join(man_dir, f"noun_table_add__{tag}.json"), "w"),
                  ensure_ascii=False, indent=2, sort_keys=True)
    print(f"  noun table: {len(noun_table)} entries (+{len(new_nouns)} new)", flush=True)

    # ---- pooled per-angle proportions + keyword agreement ----
    curves = defaultdict(dict)
    agree = Counter()
    for (cls, rel), texts in subset.items():
        c = Counter(); c_pw = Counter()
        for t in texts:
            e = manifest.get(hashlib.sha1(t.strip().encode()).hexdigest())
            if e is None:
                continue
            lab = label_from_nouns(e["nouns"], noun_table)          # v4: other-animal wins
            lab_pw = label_percept_wins(e["nouns"], noun_table)      # v3: percept wins
            c[lab] += 1
            c_pw[lab_pw] += 1
            # keyword_label returns "bird" for its duck bucket and has no "none"
            # class, so align names before comparing.
            kw = keyword_label(t)
            kw = {"bird": "duck"}.get(kw, kw)
            lab_cmp = "other" if lab_pw == "none" else lab_pw
            agree[("match" if kw == lab_cmp else "differ", kw, lab_cmp)] += 1
        n = sum(c.values())
        n_pw = sum(c_pw.values())
        if n:
            curves[cls][rel] = {**{k: c.get(k, 0) / n for k in CLASSES},
                                **{f"pw_{k}": c_pw.get(k, 0) / max(n_pw, 1) for k in CLASSES}, "n": n}
    n_match = sum(v for (m, _, _), v in agree.items() if m == "match")
    n_tot = sum(agree.values())
    print(f"  keyword-vs-judge agreement: {n_match/max(n_tot,1):.4f} over {n_tot} texts", flush=True)

    json.dump({"tag": tag, "va_images": n_va_images, "angle_lim": args.angle_lim,
               "va_samples": args.va_samples, "harper_samples": args.harper_samples,
               "keyword_agreement": n_match / max(n_tot, 1),
               "confusion": {f"{kw}->{lab}": v for (m, kw, lab), v in agree.items()},
               "curves": {cls: [dict(rel_angle=r, **curves[cls][r]) for r in sorted(curves[cls])] for cls in curves}},
              open(os.path.join(out_root, "tables", f"{tag}.json"), "w"), indent=1)

    for cls, byrel in curves.items():
      for variant, pre in (("v4_other_wins", ""), ("v3_percept_wins", "pw_")):
          rels = sorted(byrel)
          fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111)
          for lab in CLASSES:
              ax.plot(rels, [byrel[r][pre + lab] for r in rels], label=DISPLAY[lab], color=COLORS[lab])
          ax.set_xlabel("Angle (relative to center)"); ax.set_ylabel("Probability")
          ax.set_ylim(0, 1); ax.set_xlim(min(rels), max(rels)); ax.grid(True); ax.legend()
          fig.savefig(os.path.join(out_root, "plots", f"{args.model}__{args.prompt_slug}__{cls}__{variant}.png"),
                      bbox_inches="tight", pad_inches=0.05)
          plt.close(fig)
    print("done", flush=True)
