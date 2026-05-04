#!/usr/bin/env bash
# Eval: CSGO (nanowm-l2-csgo, 4 context frames)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$0")/../common.sh"
CKPT="${CKPT:-knightnemo/nanowm-l2-csgo-100k}"
exec bash "${SCRIPT_DIR}/../eval_single_model.sh" \
    game/csgo nanowm_l2_csgo "${CKPT}" csgo \
    experiment.evaluation.scheduling_mode=full_sequence
