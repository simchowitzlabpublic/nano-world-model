# Training

End-to-end guide to training a Nano World Model: workflow, design choices (with ablation tables), and pretrained reference checkpoints for each axis.

## Setup

```bash
conda env create -f environment.yml && conda activate nanowm
```

Set data paths once (or use the `local/paths.yaml` template — see [config_system.md](config_system.md#path-configuration)):

```bash
export DATASET_DIR=/path/to/dino_wm_data    # for DINO-WM envs
export CSGO_DATA_DIR=/path/to/csgo          # for CSGO
export RT1_DATA_ROOT=/path/to/rt1_fractal   # for RT-1 (LeRobot fractal)
export RESULTS_DIR=/path/to/results         # checkpoints + logs land here
```

Training also evaluates FID/FVD periodically, which needs an i3d torchscript:

```bash
mkdir -p pretrained_models/i3d
curl -L "https://www.dropbox.com/scl/fi/c5nfs6c422nlpj880jbmh/i3d_torchscript.pt?rlkey=x5xcjsrz0818i4qxyoglp5bb8&dl=1" \
    -o pretrained_models/i3d/i3d_torchscript.pt
```

## Quick start

```bash
# CSGO — NanoWM-L/2 model
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo

# RT-1 (fractal) — main run, NanoWM-B/2
python src/main.py experiment=rt1 dataset=rt1/rt1 model=nanowm_b2

# DINO-WM PushT — NanoWM-B/2
python src/main.py experiment=dino_wm_pusht dataset=dino_wm/pusht model=nanowm_b2

# Resume from checkpoint
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo \
    experiment.resume_from_checkpoint=<path/to/ckpt>
```

`experiment.resume_from_checkpoint` resumes the full PyTorch Lightning trainer
state during training (model, optimizer, scheduler, epoch/global step). For
weight-only initialization or finetuning, use `experiment.pretrained=<path>`.

Example scripts for the runs in the tables are provided
under `src/scripts/train/`.

Outputs land under `${RESULTS_DIR}/<run_dir>/`:
```
<run_dir>/
├── .hydra/                 # composed config snapshot
├── checkpoints/
│   ├── across_timesteps/   # periodic saves (every 10k steps)
│   └── latest/             # latest (overwritten every 1k steps)
└── tb/                     # tensorboard logs
```

Monitor with `tensorboard --logdir ${RESULTS_DIR}/<run_dir>/tb`, or set `wandb.enabled=true` (and `WANDB_ENTITY` / `WANDB_PROJECT`).

<div align="center">

![WandB training and validation dashboard](../assets/wandb.png)

</div>

## Training loop, in one paragraph

PyTorch Lightning drives the loop. Each step samples a `[B, T, 3, H, W]` clip, encodes frames to VAE latents, samples per-frame diffusion timesteps (logit-normal by default, SD3-style), denoises with the NanoWM transformer, computes the prediction-target loss (v / x / ε / flow), and steps the optimizer (AdamW, lr=1e-4, warmup=1000, cosine decay). Validation runs every `val_every_n_steps` (default 1k); FID/FVD every `metrics.log_every_n_train_steps` (default 5k). Checkpoints save to `latest/` every 1k steps and to `across_timesteps/` every 10k.

Knobs: `experiment.training.{batch_size, max_steps, gradient_clip_norm}`, `experiment.diffusion.{pred_name, noise_schedule, zero_terminal_snr, snr_gamma, timestep_sampling}`, `experiment.infra.{mixed_precision, num_workers, compile}`. See [config_system.md](config_system.md) for the full reference.

## Design choices

We ablate three orthogonal axes head-to-head on RT-1 and ship a checkpoint per arm. All arms share a common reference (NanoWM-B/2 · pred-v · additive injection · cosine + ZTSNR · 50k steps) and vary exactly one axis at a time. The wins inform the defaults baked into `experiment=default`.

### 1. Prediction target

We support **ε** (epsilon), **v**, **x**, and **flow-matching (with v)** prediction. Each arm runs in its native schedule (cosine + ZTSNR for v / x; linear, no ZTSNR for ε — cosine + ε is numerically degenerate at `t=T`), so the comparison isolates the prediction target rather than handicapping any one of them.

<div align="center">

| Target | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | Schedule | HF checkpoint |
|:-------|:-------|:-------|:--------|:------|:---------|:--------------|
| **v** | 23.07 | 0.760 | 0.207 | 42.27 | cosine + ZTSNR | [nanowm-b2-rt1-abl-pred-v-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-pred-v-50k) |
| x | 23.37 | **0.783** | **0.184** | 42.99 | cosine + ZTSNR | [nanowm-b2-rt1-abl-pred-x-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-pred-x-50k) |
| ε | 21.89 | 0.739 | 0.225 | 48.86 | linear | [nanowm-b2-rt1-abl-pred-epsilon-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-pred-epsilon-50k) |
| flow | **23.54** | 0.772 | 0.192 | **38.10** | cosine, no ZTSNR | [nanowm-b2-rt1-abl-pred-flow-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-pred-flow-50k) |

</div>

**Default**: `pred_name=v`, `noise_schedule=squaredcos_cap_v2`, `zero_terminal_snr=true`. Flow gives the best FID/PSNR in this sweep, while x gives the best SSIM/LPIPS; all three non-ε targets beat ε meaningfully.

```bash
# Override via CLI
python src/main.py experiment=ablation_rt1 dataset=rt1/rt1 model=nanowm_b2 \
    experiment.diffusion.pred_name=x
```

Flow matching can be launched the same way:

```bash
python src/main.py experiment=ablation_rt1 dataset=rt1/rt1 model=nanowm_b2 \
    experiment.diffusion.pred_name=flow \
    experiment.diffusion.snr_gamma=0.0 \
    experiment.diffusion.zero_terminal_snr=false
```

### 2. Action injection

Five conditioning mechanisms compared with everything else fixed. The `additive` arm coincides with the pred-v reference above (it's also the reference for every other ablation). Action embeddings come from a shared MLP; only the way they enter the transformer differs.

<div align="center">

![Action injection strategies](../assets/action_injection.png)

</div>

<div align="center">

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | Params | HF checkpoint |
|:-------|:-------|:-------|:--------|:------|:-------|:--------------|
| additive | 23.07 | 0.760 | 0.207 | 42.27 | 158.6M | [...pred-v-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-pred-v-50k) |
| adaLN | 23.19 | 0.762 | 0.206 | 43.62 | 158.6M | [...inj-adaln-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-inj-adaln-50k) |
| adaLN-fuse | 23.10 | 0.762 | 0.206 | 43.03 | 158.6M | [...inj-adaln-fuse-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-inj-adaln-fuse-50k) |
| **FiLM** | **23.20** | **0.763** | **0.203** | **40.62** | 172.8M | [...inj-film-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-inj-film-50k) |
| cross-attention | 20.82 | 0.721 | 0.242 | 51.12 | 187.0M | [...inj-cross-attention-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-inj-cross-attention-50k) |

</div>

We also ran the same five on PushT (NanoWM-B/2, 30k steps, 256 fixed val samples, seed 42):

<div align="center">

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | Extra params |
|:-------|:-------|:-------|:--------|:------|:-------------|
| **additive** | **26.20** | **0.962** | 0.053 | **23.89** | 0 |
| adaLN-fuse | 26.17 | 0.961 | **0.051** | 30.28 | 0 |
| adaLN | 26.09 | 0.960 | 0.053 | 26.32 | ~42.5M |
| cross-attention | 25.95 | 0.959 | 0.055 | 28.64 | ~28.3M |
| FiLM | 25.88 | 0.960 | 0.056 | 25.45 | ~14.4M |

</div>

**Findings**:
- The simple **additive** baseline wins on PushT with zero extra params. For low-dim actions (2D), the injection mechanism barely matters — all five land within 0.32 PSNR.
- On the higher-dim RT-1 (7D end-effector), **FiLM** edges out additive on FID; results are tighter on PSNR/SSIM/LPIPS.
- **Cross-attention** is consistently weakest at this scale.
- **Default**: `additive` — best ratio of quality to parameter count. Override with `model.action_injection.type={film,adaln,adaln_fuse,cross_attention}`.

```bash
# Use FiLM injection
python src/main.py experiment=ablation_rt1 dataset=rt1/rt1 model=nanowm_b2 \
    model.action_injection.type=film
```

### 3. Model scale

Width × depth × patch-size sweep. B/2 is the reference; S/2 is ~4× smaller, L/2 is ~3× larger.

<div align="center">

| Architecture | Params | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | HF checkpoint |
|:-------------|:-------|:-------|:-------|:--------|:------|:--------------|
| NanoWM-S/2 | 39.8M | 22.30 | 0.739 | 0.230 | 54.95 | [...scale-s2-50k](https://huggingface.co/knightnemo/nanowm-s2-rt1-abl-scale-s2-50k) |
| NanoWM-B/2 | 158.6M | 23.07 | 0.760 | 0.207 | 42.27 | [...pred-v-50k](https://huggingface.co/knightnemo/nanowm-b2-rt1-abl-pred-v-50k) |
| **NanoWM-L/2** | ~460M | **23.62** | **0.777** | **0.186** | **36.31** | [...scale-l2-50k](https://huggingface.co/knightnemo/nanowm-l2-rt1-abl-scale-l2-50k) |

</div>

Monotonic gains across all four metrics — no scaling break visible at 460M. **Default**: `nanowm_b2` (best quality / cost trade for most uses); pick L/2 when capacity matters, S/2 when iteration speed does.

```bash
python src/main.py experiment=ablation_rt1 dataset=rt1/rt1 model=nanowm_l2
```

## Reproducing the ablation arms

The completed ablation checkpoints above are all trained with `experiment=ablation_rt1` (50k steps, NanoWM-B/2 unless overridden). To reproduce e.g. the `pred=x` arm:

```bash
python src/main.py experiment=ablation_rt1 dataset=rt1/rt1 model=nanowm_b2 \
    experiment.diffusion.pred_name=x \
    experiment.diffusion.noise_schedule=squaredcos_cap_v2 \
    experiment.diffusion.zero_terminal_snr=true
```

Swap `experiment.diffusion.*` keys, `model.action_injection.type`, or `model=nanowm_l2` for the other arms.

## Pretrained checkpoints (best-config runs)

These use the winning settings (pred-v, additive, cosine + ZTSNR) on NanoWM-B/2:

<div align="center">

| Domain | Checkpoint | Steps |
|:-------|:-----------|:------|
| DINO-WM Point Maze | [nanowm-b2-dino-wm-point-maze-30k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-point-maze-30k) | 30k |
| DINO-WM Wall | [nanowm-b2-dino-wm-wall-15k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-wall-15k) | 15k |
| DINO-WM Rope | [nanowm-b2-dino-wm-rope-15k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-rope-15k) | 15k |
| DINO-WM Granular | [nanowm-b2-dino-wm-granular-15k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-granular-15k) | 15k |
| DINO-WM PushT | [nanowm-b2-dino-wm-pusht-100k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-pusht-100k) | 100k |
| RT-1 (fractal) | [nanowm-b2-rt1-300k](https://huggingface.co/knightnemo/nanowm-b2-rt1-300k) | 300k |
| CSGO | [nanowm-l2-csgo-50k](https://huggingface.co/knightnemo/nanowm-l2-csgo-50k) | 50k |
| CSGO | [nanowm-l2-csgo-100k](https://huggingface.co/knightnemo/nanowm-l2-csgo-100k) | 100k |

</div>

For evaluation numbers on these checkpoints, see [evaluation.md](evaluation.md).

## See also

- [config_system.md](config_system.md) — full Hydra config reference
- [datasets/README.md](datasets/README.md) — dataset formats and where to put files
- [evaluation.md](evaluation.md) — eval workflow + main result tables
