# Long-Horizon Rollout

Long-horizon autoregressive rollout via diffusion forcing — predict 50+ frames from a 4-frame context, sliding the context window forward as the rollout extends.

## Setup

No extras beyond the training stack:

```bash
conda env create -f environment.yml && conda activate nanowm
```

## Demo: CSGO 100k, 50-frame rollout

50-frame autoregressive rollouts (4-frame context, sequential per-frame DDIM) on held-out CSGO episodes:

<div align="center">

![3×4 CSGO long-rollout grid](../../assets/grid_video.gif)

</div>

Reproduce:

```bash
python src/sample/rollout.py \
    --config <path/to/training_run/config.yaml> \
    --ckpt <path/to/csgo_100k.ckpt> \
    --save_path results/long_rollout/csgo_100k \
    --num_samples 32 --batch_size 4 \
    --rollout_length 50 --history_length 4 \
    --num_sampling_steps 50 --scheduling_mode sequential \
    --history_stabilization_level 0.02 --fps 8
```

The `--config` path is the `.hydra/config.yaml` snapshot from the training run that produced the checkpoint — `rollout.py` needs the matching dataset / model config to decode actions and frames consistently.

## Knobs

<div align="center">

| Flag | Default | Notes |
|:-----|:--------|:------|
| `--rollout_length` | 50 | Frames to predict (must be ≥ history_length) |
| `--history_length` | 4 | Context frames at start; window slides forward as rollout extends |
| `--num_sampling_steps` | 50 | DDIM steps per frame (sequential) or per chunk (full_sequence) |
| `--scheduling_mode` | sequential | `sequential` (frame-by-frame), `full_sequence` (joint denoising), or `pyramid` |
| `--history_stabilization_level` | from config | Noise level injected into history latents (helps avoid teacher-forcing brittleness) |
| `--num_samples` | 1 | Number of starting points sampled from the val set |
| `--batch_size` | 1 | Rollouts processed in parallel |
| `--fps` | 8 | Output video frame rate |

</div>

## Outputs

```
results/long_rollout/<run>/
├── sample_0000_gen.mp4         # generated rollout
├── sample_0000_gt.mp4          # ground-truth comparison (if val frames available)
├── sample_0000_compare.mp4     # side-by-side
└── ...
```

`rollout.py` only writes videos. To compute metrics from saved rollouts, use `src/sample/evaluate_metrics.py`.

## Tips

- **Context length matters**: history_length=4 is the sweet spot for CSGO (3 FPS, ~1.3s of context). For higher-FPS datasets, increase history.
- **Noise level on history**: bumping `--history_stabilization_level` from 0.02 → 0.05 helps when the model drifts on long rollouts; too high (>0.1) starts to wash out detail.
- **DDIM step budget**: `--num_sampling_steps 50` is the quality / speed knee. Drop to 25 for quick previews; raise to 250 for top quality.
- **Native aspect ratio**: Rollout videos are saved at the configured dataset resolution. If a downstream tool needs the original frame geometry, restore it at that stage; the point-cloud script below accepts `--native_res 150 280` for CSGO.

## Going from rollout to 3D

Long-rollout videos feed naturally into the [video → 3D point cloud pipeline](video_to_3d.md). For CSGO, restore the native aspect ratio before depth inference:

```bash
python src/scripts/video_to_pointcloud.py \
    --video results/long_rollout/csgo_100k/sample_0000_gen.mp4 \
    --output output/csgo_scene.ply \
    --native_res 150 280 \
    --visualize
```

## See also

- [evaluation.md](../evaluation.md) — standardized eval (256 fixed val samples, 250 DDIM steps)
- [video_to_3d.md](video_to_3d.md) — DA3 multi-view point cloud reconstruction
- [training.md](../training.md#pretrained-checkpoints-best-config-runs) — CSGO checkpoint links (50k / 100k)
