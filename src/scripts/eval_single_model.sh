#!/usr/bin/env bash
# Evaluate a single model checkpoint.
# Usage: eval_single_model.sh <dataset> <model> <ckpt_path> <subset_name> [extra_args...]
set -euo pipefail
source "$(dirname "$0")/common.sh"
apply_gpus "${EVAL_GPUS:-0,1,2,3}"

DATASET=$1
MODEL=$2
CKPT_PATH=$3
SUBSET_NAME=$4
shift 4
EXTRA_ARGS="$@"

EVAL_DIR="${RESULTS_DIR}/eval"
SUBSET_DIR="${EVAL_DIR}/fixed_subsets"
NUM_EVAL_SAMPLES=256
EVAL_SEED=42

mkdir -p "${EVAL_DIR}" "${SUBSET_DIR}"

echo "============================================"
echo "Dataset: ${DATASET} | Model: ${MODEL}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Eval samples: ${NUM_EVAL_SAMPLES} | Seed: ${EVAL_SEED}"
echo "Start time: $(date)"
echo "============================================"

CKPT_PATH_ESCAPED="${CKPT_PATH//=/\\=}"

cd "$REPO"
exec "$PY" src/main.py \
    experiment=evaluate_only \
    model=${MODEL} \
    dataset=${DATASET} \
    dataset.loader.validation_size=null \
    dataset.loader.validation_fixed_subset_size=${NUM_EVAL_SAMPLES} \
    dataset.loader.validation_fixed_subset_seed=${EVAL_SEED} \
    "dataset.loader.validation_fixed_subset_path=${SUBSET_DIR}/${SUBSET_NAME}_val${NUM_EVAL_SAMPLES}.json" \
    "experiment.resume_from_checkpoint=${CKPT_PATH_ESCAPED}" \
    wandb.enabled=false \
    hydra.run.dir="${EVAL_DIR}/${SUBSET_NAME}" \
    ${EXTRA_ARGS}
