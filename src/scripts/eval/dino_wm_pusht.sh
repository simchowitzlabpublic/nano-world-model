#!/usr/bin/env bash
# Eval: nanowm-b2-dino-wm-pusht-100k
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$0")/../common.sh"
CKPT="${CKPT:-${RESULTS_DIR}/phase2/dino_wm_pusht/checkpoints/latest/latest-epoch=3-step=100000.ckpt}"
exec bash "${SCRIPT_DIR}/../eval_single_model.sh" \
    dino_wm/pusht nanowm_b2 "${CKPT}" dino_wm_pusht
