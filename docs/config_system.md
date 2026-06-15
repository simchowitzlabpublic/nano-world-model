# Configuration System

Nano World Model uses [Hydra](https://hydra.cc) for configuration. All training, evaluation, and planning runs go through `src/main.py`, which composes a config from defaults plus CLI overrides.

## Layout

```
src/configs/
├── config.yaml                  # Top-level — picks defaults from each group
├── experiment/                  # End-to-end run profiles (entry point)
│   ├── default.yaml             # Base training profile (training/eval/diffusion/infra)
│   ├── csgo.yaml                # CSGO main training run
│   ├── rt1.yaml                 # RT-1 main training run
│   ├── ablation_rt1.yaml        # RT-1 ablation arms (50k steps)
│   ├── dino_wm_{env}.yaml       # DINO-WM per-env (point_maze, pusht, wall, rope, granular)
│   ├── evaluate_only.yaml       # Evaluation-only run
│   └── planning.yaml            # MPC planning run
├── model/                       # Model architectures
│   ├── nanowm_s2.yaml           # NanoWM-S/2 (~40M params)
│   ├── nanowm_b2.yaml           # NanoWM-B/2 (~160M params, default)
│   ├── nanowm_l2.yaml           # NanoWM-L/2 (~460M params)
│   └── nanowm_{s2,l2}_csgo.yaml # CSGO-specific shapes (320×512 frames)
├── dataset/                     # Datasets
│   ├── dino_wm/{base,point_maze,pusht,wall,rope,granular}.yaml
│   ├── game/{base,csgo}.yaml
│   ├── rt1/{base,rt1}.yaml
│   └── lerobot/base.yaml
├── planning/
│   ├── base.yaml                # MPC + CEM defaults
│   └── planner/cem.yaml
└── local/
    └── paths.yaml.example       # Template — copy to paths.yaml (gitignored) and edit
```

## How config composition works

`src/configs/config.yaml` selects one option from each group:

```yaml
defaults:
  - model: nanowm_b2
  - latent_codec: sd_vae
  - dataset: dino_wm/point_maze
  - experiment: default
  - planning: base
```

It also resolves environment variables to fill in paths:

```yaml
dataset_dir:    ${oc.env:DATASET_DIR,./data}
csgo_data_dir:  ${oc.env:CSGO_DATA_DIR,./data/csgo}
vae_model_path: ${oc.env:VAE_MODEL_PATH,stabilityai/sd-vae-ft-mse}
webdino_model_path: ${oc.env:WEBDINO_MODEL_PATH,facebook/webssl-dino300m-full2b-224}
vjepa21_model_path: ${oc.env:VJEPA21_MODEL_PATH,null}
vjepa2_repo_path: ${oc.env:VJEPA2_REPO_PATH,null}
results_dir:    ${oc.env:RESULTS_DIR,./results}
```

Override any of these on the command line: `experiment=csgo`, `dataset=rt1/rt1`, `model=nanowm_l2`, etc. Override individual keys with `experiment.training.max_steps=100000`, `dataset.loader.validation_size=64`, etc.

## Path configuration

The codebase reads dataset/checkpoint paths from environment variables, with `./data` and `./results` as fallbacks. Two ways to configure:

**Option 1 — environment variables** (works everywhere):
```bash
export DATASET_DIR=/path/to/dino_wm_data
export CSGO_DATA_DIR=/path/to/csgo
export RT1_DATA_ROOT=/path/to/rt1_fractal
export VAE_MODEL_PATH=/path/to/vae   # or use the HF default
export WEBDINO_MODEL_PATH=facebook/webssl-dino300m-full2b-224
export VJEPA21_MODEL_PATH=/path/to/vjepa2_1_vit_large_384
export VJEPA2_REPO_PATH=/path/to/facebookresearch/vjepa2  # optional; otherwise torch.hub uses GitHub
export RESULTS_DIR=/path/to/results
```

**Option 2 — `local/paths.yaml`** (gitignored):
```bash
cp src/configs/local/paths.yaml.example src/configs/local/paths.yaml
# Edit dataset_dir / csgo_data_dir / vae_model_path / results_dir
```

CLI overrides (`dataset_dir=/path` etc.) work too and beat both.

Semantic encoder runs use the same 256-token DiT grid as SD-VAE by switching
both the codec and model shape:

| Codec | Encoder weights | Link |
|:------|:----------------|:-----|
| `webdino` | `facebook/webssl-dino300m-full2b-224` | [Hugging Face](https://huggingface.co/facebook/webssl-dino300m-full2b-224) |
| `vjepa2_1` | V-JEPA 2.1 ViT-L/16 384 checkpoint (`vjepa2_1_vitl_dist_vitG_384.pt`) | [Hugging Face collection](https://huggingface.co/collections/facebook/v-jepa-2) |

```bash
# Web-DINO: 224 input / 14px patches -> 16x16x1024 latent map
python src/main.py model=nanowm_b1_semantic latent_codec=webdino dataset=dino_wm/pusht

# V-JEPA2.1: framewise image-tokenizer path at 256 input / 16px patches -> 16x16x1024
python src/main.py model=nanowm_b1_semantic latent_codec=vjepa2_1 dataset=dino_wm/pusht \
    vjepa21_model_path=/path/to/vjepa2_1_vit_large_384 \
    vjepa2_repo_path=/path/to/facebookresearch/vjepa2
```

## Picking an experiment profile

Every run starts with `experiment=<name>`. Available profiles:

<div align="center">

| Profile | What it sets |
|:--------|:-------------|
| `default` | Base training (1M steps, lr=1e-4, bs=8, pred-v + cosine + ZTSNR) |
| `csgo` | lr=1e-5, bs=6, max_steps=50k (CSGO-specific) |
| `rt1` | RT-1 main training defaults |
| `ablation_rt1` | RT-1 ablation arms (50k steps) |
| `dino_wm_{env}` | DINO-WM env-specific overrides |
| `evaluate_only` | `tasks=[evaluate]`, full validation set |
| `planning` | `tasks=[planning]`, requires `ckpt_path=...` |

</div>

The profile composes on top of `default.yaml`, then `dataset=`, `model=`, and CLI overrides apply.

## Common override patterns

```bash
# Train CSGO with the L/2 model
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo

# Train DINO-WM PushT for 100k steps
python src/main.py experiment=dino_wm_pusht dataset=dino_wm/pusht model=nanowm_b2 \
    experiment.training.max_steps=100000

# Switch action injection (any experiment)
python src/main.py experiment=ablation_rt1 dataset=rt1/rt1 \
    model.action_injection.type=film

# Resume from a checkpoint
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo \
    experiment.resume_from_checkpoint=<path/to/ckpt>

# Disable wandb for one run
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo \
    wandb.enabled=false
```

## Key config sections

The composed config has the following top-level keys at runtime. Inspect with `--cfg job`:

```bash
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo --cfg job
```

<div align="center">

| Key | Source | Notes |
|:----|:-------|:------|
| `model.*` | `model=...` | architecture, action injection, scheduling, sampling steps |
| `dataset.*` | `dataset=...` | data paths, splits, sampling modes, action/state dims |
| `experiment.training.*` | `experiment=...` | optimizer, batch size, max_steps, checkpointing |
| `experiment.evaluation.*` | `experiment=...` | val size, FID/i3d metrics, scheduling override |
| `experiment.diffusion.*` | `experiment=...` | noise schedule, pred target, ZTSNR, snr_gamma, timestep sampler |
| `experiment.infra.*` | `experiment=...` | mixed precision, num_workers, compile, seed, num_nodes |
| `planning.*` | `planning=...` | MPC horizon, CEM samples, goal source |
| `wandb.*` | `config.yaml` + env | entity / project / mode |
| `hydra.run.dir` | `config.yaml` | output directory pattern |

</div>

## Dataset configs

Each dataset family has a `base.yaml` that fixes the schema, plus per-dataset overrides.

**DINO-WM** (`src/configs/dataset/dino_wm/`):
```yaml
# base.yaml — exhaustive train/val sampling, validation_size=32
# point_maze.yaml — frame_interval=5, action_dim=2, action_scale=1.0
# pusht.yaml      — frame_interval=5, relative actions, action_scale=100
# wall.yaml       — frame_interval=5, action_dim=2
# rope.yaml       — deformable scene
# granular.yaml   — deformable granular scene
```

**Game** (`src/configs/dataset/game/`):
```yaml
# csgo.yaml — train_slice_mode=random (5000 episodes × 1000 frames),
#             val_slice_mode=exhaustive with fixed start indices
#             action_dim=51 (keys + mouse), normalize_action=False
```

**RT-1** (`src/configs/dataset/rt1/`):
```yaml
# rt1.yaml — LeRobot HF dataset (IPEC-COMMUNITY/fractal20220817_data_lerobot),
#            train_slice_mode=random (87k episodes), action_dim=7
```

See [datasets/README.md](datasets/README.md) for the data-side details.

## Model configs

Five shipped variants:

<div align="center">

| Config | Architecture | Params | Frames | Image size |
|:-------|:-------------|:-------|:-------|:----------|
| `nanowm_s2` | NanoWM-S/2 | ~40M | 4 | 256 |
| `nanowm_b2` | NanoWM-B/2 (default) | ~160M | 4 | 256 |
| `nanowm_l2` | NanoWM-L/2 | ~460M | 4 | 256 |
| `nanowm_s2_csgo` | NanoWM-S/2 (CSGO) | ~40M | 4 | 320×512 |
| `nanowm_l2_csgo` | NanoWM-L/2 (CSGO) | ~460M | 4 | 320×512 |

</div>

Action injection is set inside the model config:
```yaml
action_injection:
  type: additive  # additive | adaln_fuse | adaln | film | cross_attention
```

See [training.md](training.md) for the design choices and ablation results.

## Debugging

```bash
# Print resolved config without running
python src/main.py experiment=csgo dataset=game/csgo --cfg job

# Print just one section
python src/main.py experiment=csgo dataset=game/csgo --cfg job --package experiment.training
```

Common errors:
- `ConfigCompositionException: Could not load experiment=foo` — typo or file doesn't exist; check `src/configs/experiment/`.
- `MissingMandatoryValue: Missing mandatory value: ckpt_path` — pass `ckpt_path=<path>` for `experiment=planning`.
- `Cannot resolve interpolation: ${oc.env:DATASET_DIR}` — set `DATASET_DIR` or pass `dataset_dir=<path>` on CLI.
