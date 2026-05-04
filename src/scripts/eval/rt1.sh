#!/usr/bin/env bash
# Eval: nanowm-b2-rt1-300k (RT-1 fractal, 300K steps)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$0")/../common.sh"
CKPT="${CKPT:-${RESULTS_DIR}/phase2/rt1/checkpoints/latest/latest-epoch=244-step=300000.ckpt}"
exec bash "${SCRIPT_DIR}/../eval_single_model.sh" \
    rt1/rt1 nanowm_b2 "${CKPT}" rt1
