# Running the experiments on a SLURM cluster

`run_array.sh` is a generic array launcher: array index *i* runs one script on
the *i*-th model of the list inside it (1–5 = the LLaVA family, 6–10 = the other
architectures). Every script is invoked as `python -m <package>.<module>` from
`src/`, so the same commands work without SLURM:

```bash
cd src && python -m experiments.rotation_beam --model llava-1.5-7b
```

Submit from the repository root (the launcher activates `.venv/` and writes
logs to `logs/`). `EXTRA` passes additional flags through to the script.

## Order

1. **Boundary fit** (prerequisite for every experiment that uses the Visual
   Anagram stimuli). `experiments.rotation_first_token` fits each model's
   duck↔rabbit decision boundary and stores it in `data/duck_rabbit/boundaries.json`.
   The file is read–merged–written by every task, so run the tasks **serially**
   (`%1`); a fit that fails the acceptance criteria leaves no entry, and the
   downstream scripts skip that model × stimulus pair with a warning.
2. The remaining experiments, in any order (GPU).
3. The CPU/API post-processing steps (LLM judge, classifiers, tables).
4. `--plot-only` replots (`replot_cpu.sh`) and the paper panels
   (`python -m figures.make_paper_panels`).

The repository ships `data/duck_rabbit/boundaries.json` as used for the paper,
so step 1 can be skipped when reproducing the paper's runs.

## GPU experiments

| Paper figure(s) | `SCRIPT=` | `--array=` | `EXTRA=` |
|---|---|---|---|
| Fig. 2, B.1/B.2 rotation; boundary fit | `experiments.rotation_first_token` | `1-10%1` | – |
| B.5 forced choice | `experiments.rotation_first_token` | `1-10` | `--prompts forced-choice --angles=-45:45:1 --center-on-boundary --results-subdir forced_choice --images harper,$(cd data/duck_rabbit && ls va_s*.png \| sed 's/.png//' \| paste -sd,)` |
| B.5 free-form descriptions (general / chain-of-thought) | `experiments.rotation_freeform` | `1-10` | – (`--chunk-idx/--num-chunks` split the stimulus set across tasks) |
| Fig. 5a, B.2 beam search (4 tokens) | `experiments.rotation_beam` | `1-10` | – |
| Fig. 5b–c, B.2/B.3/B.4/B.5 classified long continuations | `experiments.rotation_beam` | `1-10` | `--max-new-tokens 12 --num-beams 8 --prompts two-animals-pair --results-subdir beam_longcont --phase1-only` (add `--images raven_bear` for the figure–ground run) |
| B.4 Llama-3 raven–bear (24 tokens) | `experiments.rotation_beam` | `5` | `--max-new-tokens 24 --num-beams 8 --prompts two-animals-pair --results-subdir beam_longcont_llama3long --phase1-only` |
| Fig. 4, B.1/B.2 top-down cues, B.3 evidence levels, B.5 negation | `experiments.top_down_cues` | `1-10` | – |
| Fig. 3, B.1/B.2 red-circle cue | `experiments.red_circle` | `1-10` | – |
| B.4 object-count table | `experiments.object_counting` | `1-5` | – |
| B.4 raven–bear (beam, cues) | `experiments.raven_bear` | `1-5` | – |
| Fig. 6, Fig. 8, D layer-wise DLA and dominant-patch map | `experiments.layerwise_dla` | `1-5` | – |
| Fig. 7 head attention maps | `experiments.head_attention_maps` | `1-5` | – |
| Fig. 9, D bottom-up vs dominant patch | `experiments.dominant_patch_map` | `1-5` | – |
| Fig. 10, D resampling ablation | `experiments.resampling_ablation` | `1-5` (×3) | `--ablation_key resid_post` / `attn_out` / `mlp_out`, each with `--images all-valid --text_only` |
| Fig. 11, D query patching | `experiments.query_patching` | `1-3` | – (LLaVA-1.5 image geometry; reported for three models) |
| Fig. 12, D object-count vs punctuation patching | `experiments.exclusivity_patching` | `1-5` | – |

Example:

```bash
SCRIPT=experiments.rotation_first_token sbatch --array=1-10%1 slurm/run_array.sh
SCRIPT=experiments.resampling_ablation EXTRA="--ablation_key resid_post --images all-valid --text_only" sbatch --array=1-5 slurm/run_array.sh
```

Adjust `--partition`, `--mem` and the `module load` line in `run_array.sh` to
your cluster. The scripts need one GPU with ≥ 48 GB for the 13B model.

## CPU / API post-processing (no GPU; `OPENAI_API_KEY` required for the judge steps)

```bash
cd src
# 1. LLM judge of the free-form descriptions (builds the noun table used by all classifiers)
python -m analysis.llm_judge --model llava-1.5-7b --image harper --workers 8
python -m analysis.llm_judge --model Qwen2-VL-7B-Instruct --image harper --workers 8
# 2. Judge-classified free-form curves (nine models × two prompts)
for m in llava-1.5-7b llava-1.5-13b llava-v1.6-vicuna-7b llava-v1.6-mistral-7b llama3-llava-next-8b \
         Qwen2-VL-7B-Instruct smolvlm-2b idefics2-8b instructblip-vicuna-7b; do
  for s in describe-this-image describe-this-image-think-step-by-step; do
    python -m analysis.judge_freeform_curves --model $m --prompt-slug $s --workers 2
  done
done
# 3. Keyword classifier over the full free-form data (zero API cost; feeds B.5)
python -m analysis.keyword_freeform
# 4. Class shares of the long beam continuations (feeds Fig. 5b–c, B.2–B.5)
python -m analysis.classify_beam_longcont --run-dir beam_longcont
python -m analysis.classify_beam_longcont --run-dir beam_longcont_llama3long
# 5. Mean resampling heatmaps over the valid Visual Anagrams (feeds Fig. 10, App. D)
python -m analysis.aggregate_resampling
# 6. Object-count table (B.4)
python -m analysis.count_llama3_natural            # GPU: Llama-3's count readout
python -m analysis.object_count_table
# 7. Paper panels -> outputs/panels/
python -m figures.make_paper_panels
```

`replot_cpu.sh` regenerates the per-experiment plots under `outputs/*/plots`
from cached results (`--plot-only`) on a CPU partition.
