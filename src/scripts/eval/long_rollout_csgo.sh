#!/usr/bin/env bash
# Long-term rollout on CSGO using diffusion forcing with sliding window.
# 32 samples, 50 frames each, decoded to gen/gt/comparison videos.
#
# Default checkpoint follows src/scripts/train/main/csgo.sh; override with CKPT=...
set -euo pipefail
source "$(dirname "$0")/../common.sh"
apply_gpus "0"

CKPT="${CKPT:-${RESULTS_DIR}/phase2/csgo/checkpoints/latest/latest-epoch=3846-step=100000.ckpt}"
CONFIG="${CONFIG:-${RESULTS_DIR}/phase2/csgo/config.yaml}"
SAVE_PATH="${SAVE_PATH:-${RESULTS_DIR}/long_rollout/csgo_100k}"

NUM_SAMPLES="${NUM_SAMPLES:-32}"
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-50}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-50}"
SCHEDULING_MODE="${SCHEDULING_MODE:-sequential}"

mkdir -p "$SAVE_PATH"
print_launch_context "$SAVE_PATH"
cd "$REPO"
exec "$PY" src/sample/rollout.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --save_path "$SAVE_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --rollout_length "$ROLLOUT_LENGTH" \
    --history_length "$HISTORY_LENGTH" \
    --num_sampling_steps "$NUM_SAMPLING_STEPS" \
    --scheduling_mode "$SCHEDULING_MODE" \
    --history_stabilization_level 0.02 \
    --fps 8
