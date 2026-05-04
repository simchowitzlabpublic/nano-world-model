#!/usr/bin/env bash
# Eval: nanowm-b2-rt1-abl-inj-cross-attention-50k (action injection = cross_attention)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$0")/../common.sh"
CKPT="${CKPT:-${RESULTS_DIR}/ablation_rt1/inj_cross_attention/checkpoints/latest/latest-epoch=40-step=50000.ckpt}"
exec bash "${SCRIPT_DIR}/../eval_single_model.sh" \
    rt1/rt1 nanowm_b2 "${CKPT}" abl_inj_cross_attention \
    model.action_injection.type=cross_attention
