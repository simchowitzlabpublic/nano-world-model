#!/usr/bin/env bash
# RT-1 fractal main training, NanoWM-B/2.
set -euo pipefail
source "$(dirname "$0")/../../common.sh"
apply_gpus "0,1,2,3,4,5,6,7"

cd "$REPO"
RUN_DIR="${RESULTS_DIR}/phase2/rt1"
print_launch_context "$RUN_DIR"
exec "$PYTHON_BIN" src/main.py \
  dataset=rt1/rt1 \
  experiment=rt1 \
  model=nanowm_b2 \
  wandb.project="${WANDB_PROJECT}-phase2" \
  hydra.run.dir="$RUN_DIR"
