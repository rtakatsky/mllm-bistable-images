#!/bin/bash
#
# Regenerate every per-experiment plot from cached results on CPU nodes
# (no model is loaded; every experiment script supports --plot-only).
# Task index -> (script = idx / 10, model = idx % 10). Scripts that only ran
# on the LLaVA family skip the remaining model slots.
#
#   sbatch slurm/replot_cpu.sh                # everything
#   sbatch --array=1-10 slurm/replot_cpu.sh   # first script only
#
#SBATCH --job-name=bistable_replot
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-110

source .venv/bin/activate
cd src

MODELS=( \
    "llava-1.5-7b" "llava-1.5-13b" "llava-v1.6-vicuna-7b" "llava-v1.6-mistral-7b" \
    "llama3-llava-next-8b" "Qwen2-VL-2B-Instruct" "Qwen2-VL-7B-Instruct" \
    "smolvlm-2b" "idefics2-8b" "instructblip-vicuna-7b" \
)
# module : number of model slots it applies to (first N of MODELS)
SCRIPTS=( \
    "experiments.rotation_beam:10" \
    "experiments.rotation_first_token:10" \
    "experiments.rotation_freeform:10" \
    "experiments.dominant_patch_map:5" \
    "experiments.red_circle:10" \
    "experiments.top_down_cues:10" \
    "experiments.object_counting:5" \
    "experiments.raven_bear:5" \
    "experiments.exclusivity_patching:5" \
    "experiments.query_patching:3" \
    "experiments.layerwise_dla:5" \
)

idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
entry="${SCRIPTS[$(( idx / 10 ))]}"
module="${entry%%:*}"
nslots="${entry##*:}"
mi=$(( idx % 10 ))
if (( mi >= nslots )); then
    echo "[replot] ${module}: no model slot ${mi} (only ${nslots}); skipping"
    exit 0
fi
model="${MODELS[$mi]}"
echo "[replot] python -m ${module} --model ${model} --plot-only"
python -m "${module}" --model "${model}" --plot-only
