# (How) Do MLLMs Report Bistable Images Like Humans?

Code, stimuli and the stimulus-generation pipeline for the EMNLP 2026 (Main
Conference) paper *"(How) Do MLLMs Report Bistable Images Like Humans?"* by
Ryota Takatsuki, Tomoki Doi, Amane Watahiki, Anil K. Seth and Hitomi Yanaka.

Bistable images such as the duck–rabbit support several mutually incompatible
interpretations. The paper asks whether multimodal LLMs report such images the
way humans do — whether their reports are **modulable** by bottom-up visual cues
and top-down linguistic priors, and **exclusive** (committed to one
interpretation at a time) — and which internal computations support this. The
behavioural experiments run on ten MLLMs; the mechanistic analyses run on the
five LLaVA-family models.

## Repository layout

```
src/
  utils.py                 shared framework: model registry, stimulus datasets,
                           generation, decision-boundary fitting, hooks, plotting
  experiments/             GPU experiments (one script per experiment)
  stimuli/                 Visual Anagram stimulus-generation pipeline
  analysis/                CPU / API post-processing (LLM judge, classifiers, tables)
  figures/                 make_paper_panels.py — composes every paper figure
slurm/                     generic SLURM launcher + per-experiment invocations
data/duck_rabbit/          stimuli, boundaries.json, va_set_manifest.json
figures/panels/            the figure PNGs used in the paper
outputs/                   experiment results (created by the scripts, not tracked)
```

Every script is run as a module from `src/` so that `utils` and the packages
resolve without any path configuration:

```bash
cd src
python -m experiments.rotation_beam --model llava-1.5-7b
```

## Setup

Python 3.10, one GPU with ≥ 48 GB memory for the 13B model (≥ 24 GB for the
others).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are downloaded from the Hugging Face Hub on first use into
`huggingface_cache/` inside the repository. Set `VLM_BISTABLE_HF_CACHE_DIR` (or
`HF_HOME`) to put the cache elsewhere.

Environment variables used by some steps:

| Variable | Needed by |
|---|---|
| `HF_TOKEN` | `stimuli.generate_va_candidates` (DeepFloyd IF is a gated model) |
| `OPENAI_API_KEY` | `stimuli.judge_va_stimuli`, `stimuli.generate_split_controls`, `analysis.llm_judge`, `analysis.judge_freeform_curves`, `analysis.classify_beam_longcont` |

### Models

| `--model` key | Hugging Face id | Group |
|---|---|---|
| `llava-1.5-7b` | `llava-hf/llava-1.5-7b-hf` | LLaVA (behavioural + mechanistic) |
| `llava-1.5-13b` | `llava-hf/llava-1.5-13b-hf` | LLaVA |
| `llava-v1.6-vicuna-7b` | `llava-hf/llava-v1.6-vicuna-7b-hf` | LLaVA |
| `llava-v1.6-mistral-7b` | `llava-hf/llava-v1.6-mistral-7b-hf` | LLaVA |
| `llama3-llava-next-8b` | `llava-hf/llama3-llava-next-8b-hf` | LLaVA |
| `Qwen2-VL-2B-Instruct` | `Qwen/Qwen2-VL-2B-Instruct` | other (behavioural only) |
| `Qwen2-VL-7B-Instruct` | `Qwen/Qwen2-VL-7B-Instruct` | other |
| `smolvlm-2b` | `HuggingFaceTB/SmolVLM-Instruct` | other |
| `idefics2-8b` | `HuggingFaceM4/idefics2-8b` | other |
| `instructblip-vicuna-7b` | `Salesforce/instructblip-vicuna-7b` | other |

The registry is `VLM_DICT` in `src/utils.py`. All models are loaded with
`transformers==4.48.0`; `utils.load_vlm` applies a small patch to Qwen2-VL's
beam-search input expansion, which is broken in that release.

## Data

`data/duck_rabbit/` contains

- `harper.png` — the canonical duck–rabbit (Harper's version);
- `va_s<seed>.png` — the 50 synthetic Visual Anagram stimuli (RGBA, background
  removed, pre-rotated by −45° so the duck↔rabbit flip spans roughly ±45°);
- `split_s<seed>.png` — style-matched non-ambiguous controls showing a duck and
  a rabbit side by side, and `duck_s<seed>.png` / `rabbit_s<seed>.png`, the
  single-object controls derived from them;
- `raven_bear.png` — the figure–ground image of Appendix B.4;
- `va_set_manifest.json` — generation settings and accepted seeds;
  (the Visual Anagram stimuli were generated with DeepFloyd IF — *DeepFloyd is
  licensed under the DeepFloyd License, Copyright (c) Stability AI Ltd. All
  Rights Reserved.* — and are provided for research use);
- `boundaries.json` — the fitted decision boundaries (see below).

Rotations and red-circle cues are generated in memory by
`utils.RotatedRGBAImageDataset` / `utils.RedCircleImageDataset`; there are no
pre-rotated files.

**`boundaries.json` is the hub of the pipeline.** For each (model, stimulus,
prompt) it stores the rotation angle at which the model's first-token report
flips between the duck and the rabbit token, fitted by
`experiments.rotation_first_token` (`utils.decision_boundary_from_topk`). Fits
that do not meet the acceptance criteria (enough points on both sides of the
50 % line, a sufficient probability range, a monotone slope) are stored as
`null`. Every other experiment centres each Visual Anagram on its model-specific
boundary (`utils.centered_va_image`) and **skips** stimuli without a valid
boundary, printing a `[WARN] ... boundary not found` line; the number of valid
stimuli per model is reported in Appendix A of the paper. The shipped
`boundaries.json` is the one used for the paper.

## Running the experiments

The order and the exact flags of every run are listed in
[`slurm/README.md`](slurm/README.md), together with a generic SLURM array
launcher. In short:

1. `experiments.rotation_first_token` for all models (serially — it rewrites
   `boundaries.json`), or skip this step and use the shipped boundaries;
2. the other GPU experiments (`experiments.*`), in any order;
3. the CPU/API post-processing (`analysis.*`): LLM judge → judge-classified and
   keyword-classified free-form curves → beam class shares → resampling means →
   object-count table;
4. `python -m figures.make_paper_panels` to compose the figures into
   `outputs/panels/`.

Results land in `outputs/<experiment>/{text,probs,plots,tables,cache}/`. Every
experiment script accepts `--plot-only`, which regenerates its plots from the
cached results without loading a model (`slurm/replot_cpu.sh`). Existing
per-stimulus results are skipped unless `--overwrite` is given.

| Module | Paper |
|---|---|
| `experiments.rotation_first_token` | Fig. 2, App. B.1–B.2 (first-token report along rotation); forced choice, App. B.5 |
| `experiments.rotation_freeform` + `analysis.llm_judge`, `analysis.judge_freeform_curves`, `analysis.keyword_freeform` | App. B.5 (free-form descriptions, general and chain-of-thought) |
| `experiments.rotation_beam` + `analysis.classify_beam_longcont` | Fig. 5, App. B.2–B.5 (exclusivity of beam-search continuations) |
| `experiments.red_circle` | Fig. 3, App. B.1–B.2 (local visual cue) |
| `experiments.top_down_cues` | Fig. 4, App. B.1–B.3, B.5 (linguistic cues, evidence levels, negation) |
| `experiments.raven_bear`, `experiments.object_counting` + `analysis.count_llama3_natural`, `analysis.object_count_table` | App. B.4 (figure–ground boundary condition) |
| `experiments.layerwise_dla` | Fig. 6, Fig. 8, App. D (layer-wise direct logit attribution, dominant-patch map) |
| `experiments.head_attention_maps` | Fig. 7 (attention maps of the top heads) |
| `experiments.dominant_patch_map` | Fig. 9, App. D (bottom-up cue vs dominant patch) |
| `experiments.resampling_ablation` + `analysis.aggregate_resampling` | Fig. 10, App. D (resampling ablation of the top-down cue) |
| `experiments.query_patching` | Fig. 11, App. D (image-attention effect of the top-down cue) |
| `experiments.exclusivity_patching` | Fig. 12, App. D (object count vs punctuation patching) |

Notes on scope:

- The mechanistic scripts assume the LLaVA layout
  (`model.language_model.model.layers`); `query_patching` additionally assumes
  the LLaVA-1.5 image-token geometry and is reported for three models.
- `head_attention_maps` runs on the canonical duck–rabbit only.
- Fig. 3(b) of the paper shows a human fixation-density map from Hsu & Chen
  (2025), which is not redistributed here; `figures.make_paper_panels` leaves
  that panel empty unless the file is placed at
  `figures/external/hsu_chen_duck_rabbit.png`.
- Fig. 1 (schematic) was drawn by hand and is not produced by the code.

## Stimulus-generation pipeline

The Visual Anagram stimuli were produced by `src/stimuli/` (details in
Appendix A of the paper):

```bash
cd src
python -m stimuli.generate_va_candidates --seed-start 0 --seed-end 100   # DeepFloyd IF, GPU, HF_TOKEN
python -m stimuli.remove_va_background                                    # rembg (u2net), rotate −45°, crop
python -m stimuli.judge_va_stimuli --mode candidates                      # GPT judge: keep clean duck↔rabbit flips
python -m stimuli.generate_split_controls                                 # style-matched duck+rabbit controls (OpenAI image API)
python -m stimuli.make_singles_from_splits                                # duck-only / rabbit-only controls
python -m stimuli.judge_va_stimuli --mode singles
```

The accepted seeds and every setting are recorded in
`data/duck_rabbit/va_set_manifest.json`.

## Citation

```bibtex
@inproceedings{takatsuki2026bistable,
  title     = {(How) Do {MLLM}s Report Bistable Images Like Humans?},
  author    = {Takatsuki, Ryota and Doi, Tomoki and Watahiki, Amane and Seth, Anil K. and Yanaka, Hitomi},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

## Contact

Ryota Takatsuki — R.Takatsuki@sussex.ac.uk
