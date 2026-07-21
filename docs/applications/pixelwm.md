# PixelWM

PixelWM uses the existing NanoWM diffusion pipeline directly on normalized RGB. The `pixel` latent codec is an identity mapping: it has no weights, preserves raw normalized values, and exposes a decoder so the standard pixel metrics and video callbacks remain enabled. The initial `pixelwm_b16` model is NanoWM-B/16 (hidden size 768, depth 12, 12 heads) over four 256×256 RGB frames, producing 16×16 = 256 spatial tokens per frame. The presets use Gaussian diffusion with `pred_name: x`.

## Train

Always pair a PixelWM experiment preset with its explicit dataset:

```bash
python src/main.py experiment=pixelwm_rt1 dataset=rt1/rt1
python src/main.py experiment=pixelwm_point_maze dataset=dino_wm/point_maze
python src/main.py experiment=pixelwm_pusht dataset=dino_wm/pusht
```

The presets inherit the corresponding RT-1, PointMaze, and PushT batch size, worker, optimization schedule, and step budget. Pixel-space activations consume substantially more memory than compressed latents; profile launch-time batch size and accumulation overrides on the target hardware.

## Evaluate

```bash
python src/main.py experiment=pixelwm_pusht dataset=dino_wm/pusht \
    experiment.tasks='[evaluate]' experiment.resume_from_checkpoint=<ckpt> \
    dataset.loader.validation_fixed_subset_size=256 \
    dataset.loader.validation_fixed_subset_seed=42
```

## Standalone action-conditioned rollout

Use the resolved `.hydra/config.yaml` saved by training. The rollout builds the codec in that config; `--vae_model_path` remains available only as a legacy SD-VAE path override.

```bash
python src/sample/rollout.py --config <run>/.hydra/config.yaml --ckpt <ckpt> \
    --save_path results/pixelwm_rollout --rollout_length 50 --history_length 4 \
    --num_sampling_steps 50 --scheduling_mode sequential
```

Diffusion and codec outputs stay unclipped. Only MP4 writing maps normalized `[-1, 1]` RGB to `[0, 1]` and clamps it. Generated videos can be passed directly to the [video-to-3D pipeline](video_to_3d.md):

```bash
python src/scripts/video_to_pointcloud.py \
    --video results/pixelwm_rollout/sample_0000_gen.mp4 \
    --output output/pixelwm_scene.ply
```

## Planning

Planning uses the existing final-frame pixel-MSE objective. The checkpoint must remain beside its resolved training configuration (or in a directory from which planning can discover `config.yaml`). The explicit current-command model and codec settings supply runtime frame and scheduling fields and must match the checkpoint config.

```bash
# PointMaze
python src/main.py experiment=planning dataset=dino_wm/point_maze \
    model=pixelwm_b16 latent_codec=pixel ckpt_path=<ckpt> \
    planning.env_name=point_maze model.scheduling_mode=full_sequence

# PushT
python src/main.py experiment=planning dataset=dino_wm/pusht \
    model=pixelwm_b16 latent_codec=pixel ckpt_path=<ckpt> \
    planning.env_name=pusht model.scheduling_mode=full_sequence
```

Raw planning rollouts are not clamped. Pixel candidates can make CEM memory-intensive; profile `planning.cem.num_samples`, optimization iterations, sampling steps, evaluation count, and rollout batch size before a full run.

Before submitting any training, evaluation, rollout, or planning job, use a detached worktree pinned to the exact implementation commit, external dataset/results paths, and recorded experiment SHA as described in the repository experiment workflow.
