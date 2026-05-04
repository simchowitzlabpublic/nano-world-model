#!/usr/bin/env bash
# Planning (MPC) on pusht (DINO-WM aligned: horizon=5, replan_every=5).
set -euo pipefail
source "$(dirname "$0")/../common.sh"
apply_gpus "0"

CKPT="${CKPT:-${RESULTS_DIR}/phase2/dino_wm_pusht/checkpoints/latest/latest-epoch=3-step=100000.ckpt}"
# Hydra parses '=' as override delimiter; escape '=' inside values with backslash.
CKPT_ESCAPED="${CKPT//=/\\=}"
RUN_DIR="${RESULTS_DIR}/planning/pusht"

cd "$REPO"
# pusht: frame_interval=5, action_dim=2 → max_episode_steps=20 == 100 env steps.
# DINO-WM's plan_pusht.yaml uses goal_source='dset' (init/goal sampled from a
# validation trajectory; goal is reachable in goal_H planner steps by replaying
# ground-truth actions). Random pusht goals are typically unreachable in 5 steps.
exec "$PY" src/main.py \
    experiment=planning \
    model=nanowm_b2 \
    dataset=dino_wm/pusht \
    "ckpt_path=${CKPT_ESCAPED}" \
    planning.env_name=pusht \
    planning.goal_source=dset \
    planning.goal_H=5 \
    planning.horizon=5 \
    planning.replan_every=5 \
    planning.max_episode_steps=20 \
    planning.n_evals=50 \
    planning.n_plot_samples=5 \
    model.scheduling_mode=full_sequence \
    wandb.enabled=false \
    hydra.run.dir="$RUN_DIR"
