# Per-angle classification of long-continuation beams and
# mass-weighted class-share curves (replaces the string-tracking two-animals
# aggregates). Judge = the approved v3 two-step pipeline, reused verbatim:
#   step 1  LIST_PROMPT  ("List every animal in the text, each in one word.")
#   step 2  NOUN_PROMPT  (bird / rabbit / other, no examples), shared noun table
#   step 3  v4 aggregation:
#           percept label: both = bird & rabbit present; bird / rabbit only if
#           nothing else is listed; other = any other-animal present; none = empty
#           exclusivity:   n distinct animals -> single (1) / multiple (>=2) / none (0)
# normalized within the returned beams per angle; coverage (raw returned mass)
# is reported alongside. Regex classifier kept only as an agreement check.
import argparse, glob, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from importlib import import_module
from utils import PROJECT_ROOT
_v3 = import_module("analysis.llm_judge")
LIST_PROMPT, NOUN_PROMPT, PROMPT_VERSION, parse_nouns, ask_once = (
    _v3.LIST_PROMPT, _v3.NOUN_PROMPT, _v3.PROMPT_VERSION, _v3.parse_nouns, _v3.ask_once)

ANIMAL = r"(duck|ducks|bird|birds|goose|geese|swan|rabbit|rabbits|bunny|bunnies|hare|cat|dog|fox|fish|deer|animal|creature)"
ENUM = re.compile(r"\b(and|,)\s*(a |an |the )?" + ANIMAL, re.I)


def regex_multi(text):
    return bool(ENUM.search(text.lower()))


# The judge must see prefix + continuation: a continuation like " and a rabbit"
# under the prefix "I see a bird" omits the prefixed animal. Prefixes are
# recovered from the prompt slug (the JSONs store only the slug).
PREFIX_BY_SLUG_SUFFIX = [("-i-see-a-bird", "I see a bird"), ("-i-see-a-rabbit", "I see a rabbit"),
                         ("-i-see-a", "I see a"), ("-i-see", "I see"), ("-there-s-a", "There's a")]


def prefix_for_slug(slug):
    for suf, pre in PREFIX_BY_SLUG_SUFFIX:
        if slug.endswith(suf):
            return pre
    raise ValueError(f"unknown prompt prefix for slug {slug!r}; extend PREFIX_BY_SLUG_SUFFIX")


def judged_text(slug, continuation):
    return (prefix_for_slug(slug) + " " + continuation.strip()).strip()


def aggregate_v4(nouns, noun_table, cls=None):
    classes = [noun_table.get(n) for n in nouns if noun_table.get(n) in ("bird", "rabbit", "other")]
    s = set(classes)
    if cls == "raven_bear":
        # Figure-ground stimulus: the two percepts are the bird and the mammal
        # reading (bear / dog / panda / polar bear, all judged "other"), so
        # "both" = bird-class and other-class nouns co-occur; "other" = mammal
        # reading only. Duck--rabbit stimuli keep the
        # bird/rabbit definition below.
        n = len(set(nouns))
        if not s:
            return "none", "none"
        percept = ("both" if {"bird", "other"} <= s else
                   "bird" if s == {"bird"} else
                   "other" if s == {"other"} else "rabbit")
        return percept, ("single" if n == 1 else "multiple")
    if not s:
        percept = "none"
    elif "bird" in s and "rabbit" in s:
        percept = "both"
    elif "other" in s:
        percept = "other"
    elif s == {"bird"}:
        percept = "bird"
    else:
        percept = "rabbit"
    n = len(set(nouns))
    excl = "none" if n == 0 else ("single" if n == 1 else "multiple")
    return percept, excl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="beam_longcont", help="outputs/<dir> with text/<model>/<image>/<slug>.json")
    ap.add_argument("--judge-model", default="gpt-5-nano-2025-08-07")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = PROJECT_ROOT
    run = os.path.join(root, "outputs", args.run_dir)
    man_dir = os.path.join(root, "outputs", "llm_judge", "manifests", f"{args.judge_model}__{PROMPT_VERSION}")
    os.makedirs(man_dir, exist_ok=True)
    man_path = os.path.join(man_dir, f"beams__{args.run_dir}.json")
    noun_table_path = os.path.join(man_dir, "noun_table.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    noun_table = json.load(open(noun_table_path)) if os.path.exists(noun_table_path) else {}
    tab_dir, plot_dir = os.path.join(run, "tables"), os.path.join(run, "plots")
    os.makedirs(tab_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(run, "text", "*", "*", "*.json")))
    uniq = {}
    for f in files:
        slug = os.path.basename(f)[:-5]
        for item in json.load(open(f)):
            for o in item["output"]:
                for s in o["tokens"]:
                    jt = judged_text(slug, s)
                    uniq.setdefault(hashlib.sha1(jt.encode()).hexdigest(), jt)
    todo = [(k, s) for k, s in uniq.items() if k not in manifest]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(files)} files, {len(uniq)} unique beam strings, {len(manifest)} cached, {len(todo)} to judge", flush=True)

    from openai import OpenAI
    client = OpenAI()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ask_once, client, args.judge_model, LIST_PROMPT.format(text=s)): k for k, s in todo}
        for i, (fut, k) in enumerate(futs.items()):
            raw = fut.result(); manifest[k] = {"raw": raw, "nouns": parse_nouns(raw)}
            if (i + 1) % 200 == 0:
                json.dump(manifest, open(man_path, "w"), ensure_ascii=False); print(f"  judged {i+1}/{len(todo)}", flush=True)
    json.dump(manifest, open(man_path, "w"), ensure_ascii=False)
    new_nouns = sorted({n for v in manifest.values() for n in v["nouns"]} - set(noun_table))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ask_once, client, args.judge_model, NOUN_PROMPT.format(noun=n)): n for n in new_nouns}
        for fut, n in futs.items():
            raw = fut.result().strip().lower(); c = raw.split()[0].strip(".,\"'") if raw else "unparsed"
            noun_table[n] = c if c in ("bird", "rabbit", "other") else "unparsed"
    json.dump(noun_table, open(noun_table_path, "w"), ensure_ascii=False, indent=2, sort_keys=True)
    print(f"noun table: {len(noun_table)} entries ({len(new_nouns)} new: {new_nouns})", flush=True)

    # ---- per-angle mass-weighted shares ----
    PERCEPT = ("bird", "rabbit", "both", "other", "none"); EXCL = ("single", "multiple", "none")
    curves = defaultdict(lambda: defaultdict(list))  # (model,slug,cls) -> rel_angle -> list of per-image dicts
    agree = Counter()
    for f in files:
        model = f.split(os.sep)[-3]; image = f.split(os.sep)[-2]; slug = os.path.basename(f)[:-5]
        cls = next((c for c in ("va", "split", "duck", "rabbit") if image.startswith(f"{c}_s")), image)
        data = json.load(open(f)); angles = [it["angle"] for it in data]; center = angles[len(angles) // 2] if cls in ("va",) else 0
        for it in data:
            o = it["output"][0]; w = np.array(o["probs"], dtype=float)
            # EOS-excluded probabilities are PREFIX probabilities: if beam A's text is a
            # strict prefix of beam B's, A's mass contains B's. Make returned beams
            # disjoint by subtracting each longer beam's mass from its prefix beam(s).
            texts = [s.strip() for s in o["tokens"]]
            w_adj = w.copy()
            for i_a, ta in enumerate(texts):
                for i_b, tb in enumerate(texts):
                    if i_a != i_b and len(tb) > len(ta) and tb.startswith(ta) and (len(ta) == 0 or not tb[len(ta)].isalnum()):
                        w_adj[i_a] -= w[i_b]
            w = np.clip(w_adj, 0.0, None); cov = float(w.sum())
            if cov <= 0: continue
            w = w / cov
            sh_p = Counter(); sh_e = Counter()
            for s, wi in zip(o["tokens"], w):
                e = manifest.get(hashlib.sha1(judged_text(slug, s).encode()).hexdigest())
                if e is None: continue
                p, x = aggregate_v4(e["nouns"], noun_table, cls)
                sh_p[p] += wi; sh_e[x] += wi
                agree[("regex_multi", regex_multi(s), x == "multiple")] += 1
            rel = it["angle"] - center
            curves[(model, slug, cls)][rel].append({**{f"p_{k}": sh_p.get(k, 0.0) for k in PERCEPT}, **{f"e_{k}": sh_e.get(k, 0.0) for k in EXCL}, "coverage": cov})

    summary = {}
    for (model, slug, cls), byangle in curves.items():
        rows = []
        for rel in sorted(byangle):
            L = byangle[rel]; row = {"rel_angle": rel, "n_images": len(L)}
            for key in list(L[0].keys()):
                vals = [d[key] for d in L]; row[key] = float(np.mean(vals)); row[key + "_std"] = float(np.std(vals))
            rows.append(row)
        summary[f"{model}|{slug}|{cls}"] = rows
        near = [r for r in rows if abs(r["rel_angle"]) <= 15]
        print(f"{model:24s} {('hint' if 'two-animals' in slug else 'base'):4s} {cls:5s} n_img={rows[0]['n_images']:3d} "
              f"|±15°: multiple={np.mean([r['e_multiple'] for r in near]):.3f} both={np.mean([r['p_both'] for r in near]):.3f} "
              f"other={np.mean([r['p_other'] for r in near]):.3f} coverage={np.mean([r['coverage'] for r in near]):.2f}")
    json.dump({"curves": summary, "regex_vs_judge_multiple": {str(k): v for k, v in agree.items()}},
              open(os.path.join(tab_dir, "class_shares.json"), "w"), indent=1)

    # ---- plots: per model, 2x2 (rows base/hint, cols va/split): exclusivity shares ----
    models = sorted({k[0] for k in curves})
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
        for r, slug in enumerate(sorted({k[1] for k in curves if k[0] == model})):
            for c, cls in enumerate(("va", "split")):
                ax = axes[r, c]; rows = summary.get(f"{model}|{slug}|{cls}")
                if not rows: ax.set_visible(False); continue
                x = [row["rel_angle"] for row in rows]
                for key, col in (("e_single", "tab:blue"), ("e_multiple", "tab:green"), ("e_none", "tab:gray"), ("p_both", "tab:purple")):
                    y = np.array([row[key] for row in rows]); sd = np.array([row[key + "_std"] for row in rows])
                    ax.plot(x, y, color=col, lw=2, label=key.replace("e_", "").replace("p_", "percept: "))
                    ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=col, alpha=0.15, lw=0)
                ax.set_title(f"{'two animals' if 'two-animals' in slug else 'base'} | {cls}_all (n={rows[0]['n_images']})", fontsize=10)
                ax.grid(True); ax.set_ylim(0, 1)
        axes[1, 0].set_xlabel("angle − boundary"); axes[1, 1].set_xlabel("angle − boundary")
        axes[0, 0].set_ylabel("share of beam mass"); axes[1, 0].set_ylabel("share of beam mass")
        axes[0, 0].legend(fontsize=8); fig.suptitle(model)
        fig.savefig(os.path.join(plot_dir, f"{model}__class_shares.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    # ---- paper-style single panels (match plot_topk_tokens: 8x4 in, grid, legend,
    #      2.0 lw mean lines, 0.18-alpha std bands) -> plots/panels/<model>__<slug>__<cls>.png ----
    panel_dir = os.path.join(plot_dir, "panels"); os.makedirs(panel_dir, exist_ok=True)
    # Colors follow utils' beam palette so these panels read as one family with
    # the harper string-tracked plots: exclusive = green (CLASS_SUM_COLORS),
    SERIES = (("e_single", "single animal", "#1b7f3b"), ("e_multiple", "multiple animals", "#8e44ad"),
              ("p_both", "both percepts", "#c71585"), ("e_none", "no animal", "gray"))
    for key, rows in summary.items():
        model, slug, cls = key.split("|")
        x = [row["rel_angle"] for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        for k, label, col in SERIES:
            # "both percepts" (bird & rabbit) is meaningless outside the duck-rabbit stimuli
            if k == "p_both" and cls not in ("va", "split", "harper"):
                continue
            y = np.array([row[k] for row in rows]); sd = np.array([row[k + "_std"] for row in rows])
            if len(rows) > 1 and rows[0]["n_images"] > 1:
                ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), alpha=0.18, color=col, linewidth=0)
            ax.plot(x, y, linewidth=2.0, label=label, color=col, marker="o" if len(x) == 1 else None)
        ax.set_xlabel("Angle"); ax.set_ylabel("Share of beam mass"); ax.set_ylim(0, 1)
        ax.set_xlim(min(x), max(x)); ax.grid(True); ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        fig.savefig(os.path.join(panel_dir, f"{model}__{slug}__{cls}.png"), bbox_inches="tight"); plt.close(fig)
    print(f"panels: {len(summary)} written to {panel_dir}", flush=True)
    print("done", flush=True)
