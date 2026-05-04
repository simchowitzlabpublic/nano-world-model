#!/usr/bin/env bash
# Eval: RT-1 flow-matching prediction target ablation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$0")/../common.sh"
CKPT="${CKPT:-${RESULTS_DIR}/ablation_rt1/pred_flow/checkpoints/latest/latest-epoch=40-step=50000.ckpt}"
exec bash "${SCRIPT_DIR}/../eval_single_model.sh" \
    rt1/rt1 nanowm_b2 "${CKPT}" abl_pred_flow \
    experiment.diffusion.pred_name=flow \
    experiment.diffusion.snr_gamma=0.0 \
    experiment.diffusion.zero_terminal_snr=false
