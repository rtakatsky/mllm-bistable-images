# Two-step LLM judge for open-ended generations.
# Design (v3):
#   Step 1 (LLM, unprimed): the experiment's own VQA prompt retargeted to text
#           ("List every animal in the text, each in one word."), plus only a
#           "none" convention. No depiction anchoring, no hedged-form clause,
#           no duck/rabbit prior. "or"-alternations simply list both nouns and
#           land in `both`, which makes any observed exclusivity conservative.
#   Step 2 (LLM, once per UNIQUE noun): classes are bird / rabbit / other —
#           "bird" (not "duck") by design, no concrete examples in
#           the prompt (the judge's own ontology; the resulting noun_table.json
#           is printed and hand-editable, re-aggregation is free).
#   Step 3 (code): empty -> none; bird-class only -> bird; rabbit-class only
#           -> rabbit; both classes -> both; only other-animals -> other;
#           percept class + other-animal -> the percept class (tie-break).
# Keyword comparison maps keyword-"duck" ([bird,duck,goose]) to judge-"bird".
# Manifests keyed by (judge model, PROMPT_VERSION); v1 primed baseline lives in
# its own manifest dir and is compared against when present.
import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import PROJECT_ROOT

PROMPT_VERSION = "v3-vqa-mirror-1"
JUDGE_REASONING_EFFORT = "minimal"
# TPM (200k/min for gpt-5-nano) is the binding limit, and each request reserves
# its *max* output budget against it. Both judge answers are a word or a short
# comma list, so capping the budget cuts the per-request TPM charge several-fold
# and multiplies throughput.
JUDGE_MAX_OUTPUT_TOKENS = 32  # see ask_once; set to None-equivalent by editing here

LIST_PROMPT = """List every animal in the text, each in one word. If there is no animal, answer "none".

Text:
<<<
{text}
>>>"""

NOUN_PROMPT = """Answer with exactly one word: bird, rabbit, or other.
Is a "{noun}" bird-like, rabbit-like, or another kind of animal?"""

KEYWORD_MAPPING = {"duck": ["bird", "duck", "goose"], "rabbit": ["rabbit", "bunny"]}
LABELS4 = ("bird", "rabbit", "both", "other")


def classify_sentence_keyword_matching(text, mapping=KEYWORD_MAPPING):
    hits = {label for label, kws in mapping.items() if any(kw in text.lower() for kw in kws)}
    if hits == {"rabbit"}:
        return "rabbit"
    if hits == {"duck"}:
        return "bird"  # keyword duck-bucket ([bird,duck,goose]) aligned to judge "bird"
    if hits == {"duck", "rabbit"}:
        return "both"
    return "other"


def ask_once(client, model, prompt):
    # An EMPTY completion is never a valid verdict: both prompts require a word
    # ("none" when there is no animal), so "" means the call failed even though
    # the API returned 200 -- e.g. a reasoning model spending its whole output
    # budget on reasoning tokens. Retry empty responses like any other failure.
    # gpt-5-nano is a reasoning model: by default it spends ~320 output tokens
    # on reasoning for a one-word answer and sometimes returns empty content
    # (the failure above), while inflating TPM pressure -> 429s. Both judge
    # prompts are lookup-style, so minimal reasoning is the right setting.
    # Rate limits (429) are throughput shaping, not failures: retry them often
    # with a short jittered wait so workers keep the pipe full. Other errors use
    # exponential backoff. Empty completions are retried like failures (see above).
    # Retry policy: 429s are normal backpressure, not failures, and must not
    # consume the error budget. Structure: unbounded-in-time 429 waiting up to RL_BUDGET_S,
    # bounded retries for real errors and for empty completions.
    RL_BUDGET_S, MAX_ERRORS = 300.0, 6
    rl_waited, errors = 0.0, 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                reasoning_effort=JUDGE_REASONING_EFFORT,
                max_completion_tokens=JUDGE_MAX_OUTPUT_TOKENS)
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
            errors += 1
            if errors >= MAX_ERRORS:
                break
            print(f"[WARN] empty completion ({errors}/{MAX_ERRORS}); retrying", flush=True)
            time.sleep(1 + random.random())
        except Exception as e:
            if type(e).__name__ == "RateLimitError":
                if rl_waited >= RL_BUDGET_S:
                    print(f"[WARN] rate-limited for {rl_waited:.0f}s; giving up on this text", flush=True)
                    break
                w = 0.7 + 1.3 * random.random()
                rl_waited += w
                time.sleep(w)
                continue
            errors += 1
            if errors >= MAX_ERRORS:
                break
            print(f"[WARN] API error ({type(e).__name__}) {errors}/{MAX_ERRORS}; retrying", flush=True)
            time.sleep(min(2 ** errors, 30) * (0.5 + random.random()))
    print("[ERROR] giving up on one text (no usable completion)", flush=True)
    return ""


def parse_nouns(raw):
    s = raw.strip().lower()
    if not s or s == "none" or s.startswith("none") or s.startswith("no animal"):
        return []
    nouns = []
    for part in s.replace("\n", ",").split(","):
        w = part.strip().strip(".;:\"'").split()
        if not w:
            continue
        n = w[-1]  # "a bird" -> "bird"
        if n and n != "none" and n.isalpha():
            nouns.append(n)
    return sorted(set(nouns))


def aggregate(noun_classes):
    s = set(noun_classes)
    if not s:
        return "none"
    if "bird" in s and "rabbit" in s:
        return "both"
    if "bird" in s:
        return "bird"    # percept class wins over co-mentioned other-animals
    if "rabbit" in s:
        return "rabbit"
    return "other"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.5-7b")
    parser.add_argument("--image", type=str, default="harper")
    parser.add_argument("--prompt-slug", type=str, default="describe-this-image")
    parser.add_argument("--judge-model", type=str, default="gpt-5-nano-2025-08-07")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = PROJECT_ROOT
    in_path = os.path.join(root, "outputs", "rotation_freeform", "text",
                           args.model, args.image, f"{args.prompt_slug}.json")
    out_root = os.path.join(root, "outputs", "llm_judge")
    man_dir = os.path.join(out_root, "manifests", f"{args.judge_model}__{PROMPT_VERSION}")
    tab_dir = os.path.join(out_root, "tables")
    plot_dir = os.path.join(out_root, "plots")
    for d in (man_dir, tab_dir, plot_dir):
        os.makedirs(d, exist_ok=True)
    tag = f"{args.model}__{args.image}__{args.prompt_slug}"
    man_path = os.path.join(man_dir, f"{tag}.json")
    noun_table_path = os.path.join(man_dir, "noun_table.json")

    with open(in_path) as f:
        data = json.load(f)

    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    noun_table = json.load(open(noun_table_path)) if os.path.exists(noun_table_path) else {}

    uniq = {}
    for item in data:
        for t in item["output"]:
            uniq.setdefault(hashlib.sha1(t.strip().encode()).hexdigest(), t.strip())
    todo = [(k, t) for k, t in uniq.items() if k not in manifest]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{tag} [{PROMPT_VERSION}]: {sum(len(i['output']) for i in data)} generations, "
          f"{len(uniq)} unique, {len(manifest)} cached, {len(todo)} to extract", flush=True)

    from openai import OpenAI
    client = OpenAI()

    # ---- step 1 ----
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(ask_once, client, args.judge_model, LIST_PROMPT.format(text=t)): k
                   for k, t in todo}
        for fut, k in futures.items():
            raw = fut.result()
            manifest[k] = {"raw": raw, "nouns": parse_nouns(raw)}
            done += 1
            if done % 200 == 0:
                with open(man_path, "w") as f:
                    json.dump(manifest, f, ensure_ascii=False)
                print(f"extracted {done}/{len(todo)}", flush=True)
    with open(man_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"step 1 done: {len(manifest)} extractions", flush=True)

    # ---- step 2 ----
    all_nouns = sorted({n for v in manifest.values() for n in v["nouns"]})
    new_nouns = [n for n in all_nouns if n not in noun_table]
    print(f"step 2: {len(all_nouns)} unique nouns, {len(new_nouns)} new", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(ask_once, client, args.judge_model, NOUN_PROMPT.format(noun=n)): n
                   for n in new_nouns}
        for fut, n in futures.items():
            raw = fut.result().strip().lower()
            cls = raw.split()[0].strip(".,\"'") if raw else "unparsed"
            noun_table[n] = cls if cls in ("bird", "rabbit", "other") else "unparsed"
    with open(noun_table_path, "w") as f:
        json.dump(noun_table, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("noun table:", json.dumps(noun_table, sort_keys=True), flush=True)

    # ---- step 3: aggregate + compare ----
    v1_path = os.path.join(out_root, "manifests", args.judge_model, f"{tag}.json")
    v1 = json.load(open(v1_path)) if os.path.exists(v1_path) else {}

    def v3_label(key):
        e = manifest.get(key)
        if e is None:
            return None
        classes = [noun_table.get(n, "unparsed") for n in e["nouns"]]
        return aggregate([c for c in classes if c in ("bird", "rabbit", "other")])

    def collapse(lbl):
        if lbl == "duck":
            return "bird"
        return lbl if lbl in ("bird", "rabbit", "both") else "other"

    conf_kw = defaultdict(Counter)
    conf_v1 = defaultdict(Counter)
    per_angle = []
    for item in sorted(data, key=lambda x: x["angle"]):
        counts = {"kw": Counter(), "v1": Counter(), "v3": Counter()}
        v3_fine = Counter()
        for t in item["output"]:
            key = hashlib.sha1(t.strip().encode()).hexdigest()
            l3 = v3_label(key)
            if l3 is None:
                continue
            kw = classify_sentence_keyword_matching(t)
            counts["kw"][kw] += 1
            counts["v3"][collapse(l3)] += 1
            v3_fine[l3] += 1
            conf_kw[kw][collapse(l3)] += 1
            j1 = v1.get(key)
            if j1 is not None:
                l1 = collapse(j1["label"])
                counts["v1"][l1] += 1
                conf_v1[l1][collapse(l3)] += 1
        row = {"angle": item["angle"]}
        for src in ("kw", "v1", "v3"):
            tot = sum(counts[src].values())
            for lbl in LABELS4:
                row[f"{src}_{lbl}"] = counts[src].get(lbl, 0) / tot if tot else None
        tot3 = sum(v3_fine.values())
        row["v3_fine_none"] = v3_fine.get("none", 0) / tot3 if tot3 else None
        row["v3_fine_other_animal"] = v3_fine.get("other", 0) / tot3 if tot3 else None
        per_angle.append(row)

    def agree(conf):
        a = sum(conf[l][l] for l in LABELS4)
        t = sum(sum(c.values()) for c in conf.values())
        return a / t if t else None

    summary = {
        "tag": tag, "prompt_version": PROMPT_VERSION, "judge_model": args.judge_model,
        "n_unique_nouns": len(all_nouns),
        "agreement_v3_vs_keyword": agree(conf_kw),
        "agreement_v3_vs_v1primed": agree(conf_v1) if v1 else None,
        "confusion_keyword_vs_v3": {k: dict(v) for k, v in conf_kw.items()},
        "confusion_v1_vs_v3": {k: dict(v) for k, v in conf_v1.items()} if v1 else None,
        "noun_table": noun_table,
    }
    with open(os.path.join(tab_dir, f"{tag}__{PROMPT_VERSION}.json"), "w") as f:
        json.dump({"summary": summary, "per_angle": per_angle}, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("confusion")}, indent=2), flush=True)

    angles = [r["angle"] for r in per_angle]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"bird": "tab:blue", "rabbit": "tab:orange", "both": "tab:green", "other": "tab:gray"}
    for lbl, c in colors.items():
        ax.plot(angles, [r[f"v3_{lbl}"] for r in per_angle], color=c, linewidth=2, label=f"{lbl} (2-step)")
        ax.plot(angles, [r[f"kw_{lbl}"] for r in per_angle], color=c, linewidth=1.2, linestyle="--", label=f"{lbl} (keyword)")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Probability")
    ax.grid(True)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(plot_dir, f"{tag}__{PROMPT_VERSION}__vs_keyword.png"), dpi=150, bbox_inches="tight")
    print("done", flush=True)
