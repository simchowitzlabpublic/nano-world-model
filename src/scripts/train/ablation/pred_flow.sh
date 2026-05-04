#!/usr/bin/env bash
# RT-1 ablation: prediction target = flow matching.
set -euo pipefail
source "$(dirname "$0")/../../common.sh"
apply_gpus "0,1,2,3,4,5,6,7"

cd "$REPO"
RUN_DIR="${RESULTS_DIR}/ablation_rt1/pred_flow"
print_launch_context "$RUN_DIR"
exec "$PYTHON_BIN" src/main.py \
  dataset=rt1/rt1 \
  model=nanowm_b2 \
  experiment=ablation_rt1 \
  experiment.diffusion.pred_name=flow \
  experiment.diffusion.snr_gamma=0.0 \
  experiment.diffusion.zero_terminal_snr=false \
  experiment.training.batch_size="${BATCH_SIZE:-8}" \
  wandb.project="${WANDB_PROJECT}-ablation" \
  hydra.run.dir="$RUN_DIR"
