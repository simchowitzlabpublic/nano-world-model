#!/usr/bin/env bash
# Shared runtime defaults for public launchers.
#
# Cluster-specific paths should be provided by environment variables rather than
# hardcoded here. Example:
#   RT1_DATA_ROOT=/path/to/rt1 RESULTS_DIR=/path/to/results bash src/scripts/train/main/rt1.sh

set -u

if [[ -z "${REPO:-}" ]]; then
  REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || { cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P; })
  export REPO
fi
export PYTHON_BIN=${PYTHON_BIN:-python}
export PY=${PY:-$PYTHON_BIN}

export PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-${REPO}/pretrained_models}
export DATASET_DIR=${DATASET_DIR:-${REPO}/data}
export CSGO_DATA_DIR=${CSGO_DATA_DIR:-${REPO}/data/csgo}
export RT1_DATA_ROOT=${RT1_DATA_ROOT:-${REPO}/data/rt1_fractal}
export RESULTS_DIR=${RESULTS_DIR:-${REPO}/results}
export VAE_MODEL_PATH=${VAE_MODEL_PATH:-stabilityai/sd-vae-ft-mse}

export WANDB_PROJECT=${WANDB_PROJECT:-nano-world-model}
export WANDB_MODE=${WANDB_MODE:-online}

apply_gpus() {
  local default="$1"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS:-$default}"
  fi
}

print_launch_context() {
  local run_dir="$1"
  cat <<EOF
[launch] repo=$REPO
[launch] python=$PYTHON_BIN
[launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}
[launch] DATASET_DIR=$DATASET_DIR
[launch] RT1_DATA_ROOT=$RT1_DATA_ROOT
[launch] CSGO_DATA_DIR=$CSGO_DATA_DIR
[launch] PRETRAINED_MODELS_DIR=$PRETRAINED_MODELS_DIR
[launch] VAE_MODEL_PATH=$VAE_MODEL_PATH
[launch] RESULTS_DIR=$RESULTS_DIR
[launch] WANDB_MODE=$WANDB_MODE
[launch] hydra.run.dir=$run_dir
EOF
}
