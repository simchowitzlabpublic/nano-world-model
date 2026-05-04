#!/usr/bin/env bash
# CSGO main training, NanoWM-L/2.
set -euo pipefail
source "$(dirname "$0")/../../common.sh"
apply_gpus "0,1,2,3,4,5,6,7"

cd "$REPO"
RUN_DIR="${RESULTS_DIR}/phase2/csgo"
print_launch_context "$RUN_DIR"
exec "$PYTHON_BIN" src/main.py \
  dataset=game/csgo \
  experiment=csgo \
  model=nanowm_l2_csgo \
  experiment.training.batch_size="${BATCH_SIZE:-4}" \
  wandb.project="${WANDB_PROJECT}-phase2" \
  hydra.run.dir="$RUN_DIR"
