# Documentation

Detailed reference for Nano World Model. The top-level [README](../README.md) covers install + a first training command; these docs go deeper.

## Layout

```
docs/
├── README.md             (this file — index)
├── config_system.md      (Hydra config: structure, overrides, paths, debugging)
├── training.md           (training workflow + design choices + ablation tables)
├── evaluation.md         (eval workflow + main result tables + sampling)
├── datasets/
│   └── README.md         (DINO-WM, RT-1, CSGO formats and configs)
└── applications/
    ├── planning.md       (MPC + CEM model-predictive control)
    ├── long_rollout.md   (long-horizon autoregressive rollout)
    └── video_to_3d.md    (Depth Anything 3 → point cloud pipeline)
```

## Index

- **[Configuration system](config_system.md)** — Hydra layout, composition, environment variables, common overrides, debugging.
- **[Training](training.md)** — workflow + the four design axes (prediction target, action injection, model scale, EMA) with ablation tables and pretrained checkpoints.
- **[Evaluation](evaluation.md)** — `experiment=evaluate_only`, scheduling modes, metric definitions, headline numbers on each domain.
- **[Datasets](datasets/README.md)** — DINO-WM (5 envs), RT-1 fractal, CSGO. Download / split / format / config.
- **[Planning](applications/planning.md)** — MPC + CEM over the diffusion world model. point_maze and PushT recipes.
- **[Long rollout](applications/long_rollout.md)** — 50-frame autoregressive rollout with sliding context window. CSGO demo.
- **[Video → 3D point cloud](applications/video_to_3d.md)** — DA3 multi-view depth + viser viewer.

## Entrypoint

All training, evaluation, and planning runs go through `src/main.py`:

```bash
# Training
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo

# Evaluation
python src/main.py experiment=evaluate_only dataset=dino_wm/pusht model=nanowm_b2 \
    experiment.resume_from_checkpoint=<path/to/checkpoint.ckpt>

# Planning
python src/main.py experiment=planning dataset=dino_wm/point_maze model=nanowm_b2 \
    ckpt_path=<path/to/checkpoint.ckpt>
```

See [config_system.md](config_system.md) for the full set of experiment / dataset / model options.
