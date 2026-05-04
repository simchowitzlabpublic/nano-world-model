#!/usr/bin/env bash
# DINO-WM Point Maze main training, NanoWM-B/2.
set -euo pipefail
source "$(dirname "$0")/../../common.sh"
apply_gpus "0,1,2,3,4,5,6,7"

cd "$REPO"
RUN_DIR="${RESULTS_DIR}/phase2/dino_wm_point_maze"
print_launch_context "$RUN_DIR"
exec "$PYTHON_BIN" src/main.py \
  dataset=dino_wm/point_maze \
  experiment=dino_wm_point_maze \
  model=nanowm_b2 \
  wandb.project="${WANDB_PROJECT}-phase2" \
  hydra.run.dir="$RUN_DIR"
