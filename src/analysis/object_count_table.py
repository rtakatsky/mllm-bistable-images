# Appendix table: one
# models x stimulus-groups table replacing the per-model object-count panels.
#   left block : first-token p("1") after "How many objects are in the image?
#                Answer with a number only." (object_counting run; " " prefix, angle 0 /
#                boundary angle for VA), mean +- SD over the group.
#   right block: judge-classified share of beam mass naming multiple animals at
#                angle 0 (boundary for VA) under the default prompt/prefix
#                (beam_longcont + classify_beam_longcont).
# split_s* rows are gated on the corresponding va_s* boundary (same convention
# as every other split aggregate). Writes outputs/object_counting/tables/object_count.tex plus a
# JSON with the numbers quoted in the text. CPU-only.
import json, os, sys
from utils import PROJECT_ROOT

ROOT = PROJECT_ROOT
MODELS = [("llava-1.5-7b", "LLaVA-1.5-7B"), ("llava-1.5-13b", "LLaVA-1.5-13B"),
          ("llava-v1.6-vicuna-7b", "LLaVA-v1.6-Vicuna-7B"), ("llava-v1.6-mistral-7b", "LLaVA-v1.6-Mistral-7B"),
          ("llama3-llava-next-8b", "Llama3-LLaVA-Next-8B")]
GROUPS = [("harper", "Harper"), ("raven_bear", "Raven--Bear"), ("va", "VA"), ("split", "Split")]
BASE_SLUG = "list-every-animal-in-the-image-each-in-one-word-i-see-a"

summ_dir = os.path.join(ROOT, "outputs", "object_counting", "tables")
# Beam-class shares come from the shared 12-token run, except Llama-3: under the
# corrected chat template it answers the default query verbosely ("I see a single
# animal in the image, which appears to be ..."), so at 12 tokens much of its beam
# mass has not named an animal yet. It was rerun at 24 tokens (identical otherwise);
# the budget changes nothing where 12 sufficed (harper 1.00/0.00, split 0.001/0.999)
# and resolves raven-bear (single 0.33 -> 1.00). See slurm/README.md (Llama-3 24-token run).
# All models share the 12-token continuation budget. The single exception is
# Llama-3 on Raven--Bear: under the corrected chat template it hedges there
# ("I see a single animal in the image, which appears to be ..."), so at 12
# tokens 67% of its beam mass has not named an animal and the share is
# undefined; that cell uses its 24-token rerun (single 0.33 -> 1.00). The budget
# changes nothing where 12 tokens already sufficed (harper and split identical,
# VA single 0.927 -> 0.947), so every other cell stays on the shared run.
SHARES_RUN = {("llama3-llava-next-8b", "raven_bear"): "beam_longcont_llama3long"}
def _shares(model, group):
    run = SHARES_RUN.get((model, group), "beam_longcont")
    return json.load(open(os.path.join(ROOT, "outputs", run, "tables", "class_shares.json")))["curves"]


def mean_sd(vals):
    if not vals:
        return (float("nan"), float("nan"), 0)
    m = sum(vals) / len(vals)
    sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5  # population SD, as in the plots
    return (m, sd, len(vals))


def pair(a, b, n):
    return "--" if n == 0 else f"{a:.2f} / {b:.2f}"


SHORT = {"llava-1.5-7b": "1.5-7B", "llava-1.5-13b": "1.5-13B", "llava-v1.6-vicuna-7b": "v1.6-Vicuna",
         "llava-v1.6-mistral-7b": "v1.6-Mistral", "llama3-llava-next-8b": "Llama3-NeXT"}



def main():
    boundaries = json.load(open(os.path.join(ROOT, "data", "duck_rabbit", "boundaries.json")))
    numbers, cells, nva = {}, {}, {}
    for key, label in MODELS:
        sp = os.path.join(summ_dir, key, "summary.json")
        if not os.path.exists(sp):
            print(f"[WARN] missing {sp}; column skipped"); continue
        s = json.load(open(sp))
        # Llama-3 answers "\n1": its chat template ends the turn with the merged token
        # ĊĊ, so the " " prefix used for the SentencePiece models is off-path for it.
        # Use the same readout as Figure~45 (token-level append of Ċ, i.e. the model's
        # own answer step) where available.
        nat = os.path.join(os.path.dirname(sp), "summary_natural.json")
        if os.path.exists(nat):
            nats = json.load(open(nat))["images"]
            for k, v in s["images"].items():
                if k in nats:
                    v["first_token"] = nats[k]["first_token"]
            print(f"[{key}] count readout: model's own answer step ({len(nats)} stimuli)")
        valid_va = {k for k, v in s["images"].items() if v["group"] == "va"}

        def group_images(g):
            return [(k, v) for k, v in s["images"].items() if v["group"] == g
                    and not (g == "split" and f"va_{k.split('_')[-1]}" not in valid_va)]

        nums = {}
        for g, _ in GROUPS:
            imgs = group_images(g)
            # Reported counts are beam-search shares: they read the model's own answer
            # path and need no whitespace prefix (exact only for the SentencePiece
            # models; Llama-3 answers "\n1"). The prefixed first-token probabilities
            # are kept in the JSON for the agreement check quoted in the text.
            b1 = mean_sd([v["beam"]["shares"].get("1", 0.0) for _, v in imgs if v["beam"]])
            b2 = mean_sd([v["beam"]["shares"].get("2", 0.0) for _, v in imgs if v["beam"]])
            p1 = mean_sd([v["first_token"]["1"] for _, v in imgs])
            p2 = mean_sd([v["first_token"]["2"] for _, v in imgs])
            # First-token p("1")/p("2") after the " " prefix -- the same token
            # probability Figure~45 reads, and clean for every model (digit mass
            # >= 0.997, llama3 included). The beam shares stay in the JSON only as
            # a cross-check (do not complicate the count measure).
            cells[("count", g, key)] = pair(p1[0], p2[0], p1[2])
            nums[f"p1_{g}"], nums[f"p2_{g}"] = p1, p2
            nums[f"beam_p1_{g}"], nums[f"beam_p2_{g}"] = b1, b2
            nums[f"absdiff_{g}"] = abs(b1[0] - p1[0]) if b1[2] else float("nan")
            curve = _shares(key, g).get(f"{key}|{BASE_SLUG}|{g}")
            if curve is None:
                cells[("beam", g, key)] = "--"; nums[f"single_{g}"] = nums[f"multi_{g}"] = None
            else:
                r0 = [r for r in curve if r["rel_angle"] == 0][0]
                cells[("beam", g, key)] = pair(r0["e_single"], r0["e_multiple"], r0["n_images"])
                nums[f"single_{g}"] = (r0["e_single"], r0["e_single_std"], r0["n_images"])
                nums[f"multi_{g}"] = (r0["e_multiple"], r0["e_multiple_std"], r0["n_images"])
        nva[key] = nums["p1_va"][2]
        numbers[key] = nums

    keys = [k for k, _ in MODELS if k in numbers]
    def block(kind, title):
        out = [f"\\multicolumn{{{len(keys) + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for g, glab in GROUPS:
            out.append(f"\\quad {glab} & " + " & ".join(cells[(kind, g, k)] for k in keys) + " \\\\")
        return out

    tex = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\begin{tabular}{l" + "c" * len(keys) + "}", "\\toprule",
        "Stimulus & " + " & ".join(f"\\textit{{{SHORT[k]}}}" for k in keys) + " \\\\",
        "($n$ valid VA) & " + " & ".join(f"({nva[k]})" for k in keys) + " \\\\",
        "\\midrule",
        *block("count", "Object-count query: probability of the answer \\texttt{\"1\"} / \\texttt{\"2\"}"),
        "\\addlinespace",
        *block("beam", "Default query: share of beam mass naming one / multiple animals"),
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Object count and exclusivity across stimuli, for the five LLaVA-family models. Top block: probability "
        "mass on the answers \\texttt{\"1\"} and \\texttt{\"2\"} for \\texttt{\"How many objects are in the image? Answer with "
        "a number only.\"} under beam search (the remainder answers \"3\" or more). Bottom block: shares of beam mass whose "
        "continuation names a single vs.\\ multiple animals under the default query and prefix "
        "(Appendix~\\ref{sec:appendix_beam_classification}; the remainder names no animal). The two blocks are separate "
        "measurements: a single-object answer in the top block corresponds to exclusive reporting in the bottom one. Beam "
        "continuations use a 12-token budget, except \\textit{Llama3-NeXT} on Raven--Bear, which uses 24 tokens because "
        "at 12 tokens 67\\% of its beam mass had not yet named an animal. Harper "
        "and Raven--Bear at the original orientation; Visual Anagrams (VA) at each model\'s boundary angle and the matched "
        "split controls at $0^\\circ$, means over the model\'s valid stimuli ($n$ per column).}",
        "\\label{tab:appendix_object_count}", "\\end{table*}"]
    out_dir = os.path.join(ROOT, "outputs", "object_counting", "tables"); os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "object_count.tex"), "w").write("\n".join(tex) + "\n")
    json.dump(numbers, open(os.path.join(out_dir, "object_count_numbers.json"), "w"), indent=1)
    print("\n".join(tex))



if __name__ == "__main__":
    main()
