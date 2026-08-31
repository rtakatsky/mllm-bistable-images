#!/bin/bash
#
# Generic SLURM array launcher: one task per model.
#
#   SCRIPT=experiments.rotation_beam sbatch --array=1-10 slurm/run_array.sh
#   SCRIPT=experiments.layerwise_dla sbatch --array=1-5  slurm/run_array.sh      # LLaVA-5 only
#   SCRIPT=experiments.rotation_beam EXTRA="--results-subdir beam_longcont --phase1-only ..." sbatch --array=1-10 slurm/run_array.sh
#
# The array index selects the model from MODELS below (1-5 = LLaVA family,
# 6-10 = other architectures). See slurm/README.md for the per-experiment
# invocations. Submit from the repository root; logs go to logs/.
#
#SBATCH --job-name=bistable
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -u
: "${SCRIPT:?set SCRIPT=<package.module>, e.g. experiments.rotation_beam}"
EXTRA="${EXTRA:-}"

MODELS=( \
    "llava-1.5-7b" "llava-1.5-13b" "llava-v1.6-vicuna-7b" "llava-v1.6-mistral-7b" "llama3-llava-next-8b" \
    "Qwen2-VL-2B-Instruct" "Qwen2-VL-7B-Instruct" "smolvlm-2b" "idefics2-8b" "instructblip-vicuna-7b" \
)
idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
MODEL="${MODELS[$idx]}"

# module load python/3.10 cuda/12.1   # uncomment / adapt to your cluster
source .venv/bin/activate
cd src
echo "[run_array] python -m ${SCRIPT} --model ${MODEL} ${EXTRA}"
python -m "${SCRIPT}" --model "${MODEL}" ${EXTRA}
