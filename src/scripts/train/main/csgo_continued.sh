#!/usr/bin/env bash
# CSGO continued training: resume the phase2 CSGO run to 100k steps.
set -euo pipefail
source "$(dirname "$0")/../../common.sh"
apply_gpus "0,1,2,3,4,5,6,7"

RUN_DIR="${RESULTS_DIR}/phase2/csgo"
CKPT="${CKPT:-$(ls -t "${RUN_DIR}/checkpoints/latest/"*.ckpt 2>/dev/null | head -1)}"
if [[ -z "$CKPT" ]]; then
  echo "[abort] No checkpoint found in ${RUN_DIR}/checkpoints/latest/. Set CKPT=/path/to/checkpoint.ckpt."
  exit 1
fi
CKPT_ESCAPED="${CKPT//=/\\=}"

cd "$REPO"
print_launch_context "$RUN_DIR"
exec "$PYTHON_BIN" src/main.py \
  dataset=game/csgo \
  model=nanowm_l2_csgo \
  experiment=csgo \
  experiment.training.max_steps=100000 \
  "experiment.resume_from_checkpoint=${CKPT_ESCAPED}" \
  experiment.evaluation.scheduling_mode=full_sequence \
  wandb.project="${WANDB_PROJECT}-phase2" \
  hydra.run.dir="$RUN_DIR"
