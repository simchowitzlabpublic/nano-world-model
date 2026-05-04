#!/usr/bin/env bash
# Planning (MPC) on point_maze
set -euo pipefail
source "$(dirname "$0")/../common.sh"
apply_gpus "0"

CKPT="${CKPT:-${RESULTS_DIR}/phase2/dino_wm_point_maze/checkpoints/latest/latest-epoch=3-step=30000.ckpt}"
# Hydra parses '=' as override delimiter; escape '=' inside values with backslash.
CKPT_ESCAPED="${CKPT//=/\\=}"
RUN_DIR="${RESULTS_DIR}/planning/point_maze"

cd "$REPO"
# point_maze: frame_interval=5, action_dim=2 → max_episode_steps=10 == 50 env steps.
# We use goal_source='dset' (DINO-WM's plan_point_maze.yaml uses random_state, but switching
# to dset guarantees the goal is reachable in goal_H planner steps and avoids unsolvable
# corner cases in the success-rate measurement).
exec "$PY" src/main.py \
    experiment=planning \
    model=nanowm_b2 \
    dataset=dino_wm/point_maze \
    "ckpt_path=${CKPT_ESCAPED}" \
    planning.env_name=point_maze \
    planning.goal_source=dset \
    planning.goal_H=5 \
    planning.horizon=5 \
    planning.replan_every=5 \
    planning.max_episode_steps=10 \
    planning.n_evals=50 \
    planning.n_plot_samples=5 \
    model.scheduling_mode=full_sequence \
    wandb.enabled=false \
    hydra.run.dir="$RUN_DIR"
