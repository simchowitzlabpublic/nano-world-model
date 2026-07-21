# Planning (MPC + CEM)

CEM-style model-predictive control over the diffusion world model. Each plan samples action sequences, rolls them out through the WM's configured latent codec, scores them by MSE against a goal representation, and updates the sampling distribution toward the elites — a direct analogue of the [DINO-WM](https://github.com/gaoyuezhou/dino_wm) planning protocol but with a diffusion rollout. For identity-coded RGB commands and memory notes, see the [PixelWM guide](pixelwm.md).

## Setup

Planning needs the simulator environments on top of the training stack:

```bash
conda env create -f environment.yml && conda activate nanowm   # base stack

# Planning-specific extras (PushT, point_maze, deformable envs)
pip install gym pymunk pygame shapely scikit-image d4rl
```

**point_maze** also requires **MuJoCo 2.10** (binary, not pip):

```bash
# 1) Download MuJoCo 2.10 binary from https://github.com/google-deepmind/mujoco/releases/tag/2.1.0
#    Extract to ~/.mujoco/mujoco210
# 2) Add the runtime dir to LD_LIBRARY_PATH:
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
# 3) Verify:
python -c "from gym.envs.mujoco import mujoco_env; from d4rl import offline_env; print('OK')"
```

PushT only needs the pip extras above — no MuJoCo.

## Quick start

```bash
# point_maze: 50 episodes, dataset goals
python src/main.py experiment=planning model=nanowm_b2 dataset=dino_wm/point_maze \
    ckpt_path=<path/to/point_maze.ckpt> \
    planning.env_name=point_maze planning.goal_source=dset planning.goal_H=5 \
    planning.horizon=5 planning.replan_every=5 planning.max_episode_steps=10 \
    planning.n_evals=50 model.scheduling_mode=full_sequence

# pusht: 50 episodes, dataset goals (dataset goals = guaranteed-reachable
# target reached by replaying ground-truth actions for goal_H planner steps;
# pusht's random goals are typically not reachable in 5 planner steps)
python src/main.py experiment=planning model=nanowm_b2 dataset=dino_wm/pusht \
    ckpt_path=<path/to/pusht.ckpt> \
    planning.env_name=pusht planning.goal_source=dset planning.goal_H=5 \
    planning.horizon=5 planning.replan_every=5 planning.max_episode_steps=20 \
    planning.n_evals=50 model.scheduling_mode=full_sequence
```

## How it works

```
At each replan step:
  1. Encode current obs to latent z_0
  2. Sample N action sequences a^(i) ~ N(μ, σ²·I), shape [N, horizon, action_dim]
  3. Rollout each sequence through the WM: z_t+1 = WM(z_t, a^(i)_t) for t=0..horizon-1
  4. Score by MSE against goal latent z_goal (last frame, by default)
  5. Pick top-K elites; refit (μ, σ) to them
  6. Repeat opt_steps times → final μ is the planned action sequence
  7. Execute first replan_every actions in the env, then go back to (1)
```

Configured by `src/configs/planning/base.yaml`.

## Knobs

<div align="center">

| Group | Key | Default | Notes |
|:------|:----|:--------|:------|
| MPC | `horizon` | 5 | Planning lookahead in WM steps |
| | `replan_every` | 5 | Steps to execute before replanning (DINO-WM uses open-loop chunks: replan_every == horizon) |
| | `max_episode_steps` | 50 | Cap on env steps per episode |
| CEM | `cem.num_samples` | 100 | Action sequences sampled per iteration (DINO-WM: 300) |
| | `cem.topk` | 10 | Elites kept (DINO-WM: 30) |
| | `cem.opt_steps` | 30 | CEM iterations per replan |
| | `cem.var_scale` | 1.0 | Initial std of action distribution |
| Goal | `goal_source` | `random_state` | `random_state` (env.sample_random_init_goal_states) or `dset` (replay ground-truth actions for `goal_H` steps from a val trajectory) |
| | `goal_H` | 5 | Steps between init and goal when `goal_source=dset` |
| WM rollout | `num_sampling_steps` | 20 | DDIM steps for the in-the-loop rollout (smaller = faster, lower quality) |
| | `eta` | 0.0 | DDIM eta (0 = deterministic) |
| Objective | `objective.mode` | `last` | `last` (MSE on last predicted frame only) or `all` (mean across the horizon) |
| | `objective.alpha` | 1.0 | Power applied to MSE: `loss = MSE^alpha` |
| | `objective.base` | 2.0 | Base for elite weighting (`exp(-base * loss)`) |
| Eval | `n_evals` | 50 | Episodes |
| | `n_plot_samples` | 5 | Episodes to render to MP4 |

</div>

`goal_source=random_state` is what DINO-WM uses for point_maze and wall (random goals in the state space, not always reachable in `goal_H` steps — success rate measures both planning and feasibility). For pusht, rope, and granular, `goal_source=dset` is the right choice — random goals are typically unreachable in 5 planner steps.

## Outputs

Under `${RESULTS_DIR}/<run_dir>/planning_results/`:

```
planning_results/
├── episode_000.mp4         # rollout for episode 0 (up to n_plot_samples)
├── episode_001.mp4
├── ...
└── planning_results.json   # success_rate, state_dist_mean, per-episode metrics
```

`planning_results.json` schema:
```json
{
  "success_rate": 0.42,             // fraction of episodes hitting the goal
  "state_dist_mean": 0.087,         // mean L2 distance to goal at episode end
  "per_episode": [
    {"success": true, "state_dist": 0.012, "n_steps": 7, ...},
    ...
  ]
}
```

## Scheduling mode for planning

Set `model.scheduling_mode=full_sequence` for planning. The default `sequential` mode would denoise the WM rollout frame-by-frame, which is much slower per CEM evaluation. `full_sequence` denoises all `horizon` frames jointly with `model.num_sampling_steps` DDIM steps — fast enough to run 100 samples × 30 CEM iterations per replan.

## See also

- [training.md](../training.md) — training the WM checkpoints used here
- [config_system.md](../config_system.md#picking-an-experiment-profile) — `experiment=planning` profile details
- DINO-WM paper: Zhou et al., [DINO-WM: World Models on Pre-trained Visual Features Enable Zero-Shot Planning](https://arxiv.org/abs/2411.04983)
