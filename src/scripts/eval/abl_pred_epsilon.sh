#!/usr/bin/env bash
# Eval: nanowm-b2-rt1-abl-pred-epsilon-50k (prediction target = epsilon)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$0")/../common.sh"
CKPT="${CKPT:-${RESULTS_DIR}/ablation_rt1/pred_epsilon/checkpoints/latest/latest-epoch=40-step=50000.ckpt}"
exec bash "${SCRIPT_DIR}/../eval_single_model.sh" \
    rt1/rt1 nanowm_b2 "${CKPT}" abl_pred_epsilon \
    experiment.diffusion.pred_name=epsilon \
    experiment.diffusion.noise_schedule=linear \
    experiment.diffusion.zero_terminal_snr=false
