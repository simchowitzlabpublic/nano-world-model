<div align="center">
<h1>🌍 Nano World Model</h1>
</div>

<div align="center">
<a href='https://huggingface.co/collections/knightnemo/nano-world-model'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Page-blue'></a>
<a href='https://simchowitzlabpublic.github.io/nano-world-model/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</div>

A minimalist repository for training video world models based on diffusion-forcing.

<div align="center">

![3×4 rollout grid](assets/grid_video.gif)

</div>

## Key Features

- 🚀 **Instant Start** — Minimal dependencies, easy data loading. From clone to first rollout in minutes.
- 🛠️ **Unified Pipeline** — Training, Validation, Evaluation; All managed with clean hydra-based configuration systems.
- 🔬 **Scientific Transparency** — Clean codebase with head-to-head ablations across prediction target, action injection, and model scale; Fully open-source, including model checkpoints.
- 🤖 **Diverse Applications** — Long-horizon rollouts, rollout to 3d point clouds, planning (MPC) out of the box.

## 🚀 Quick Start

```bash
git clone https://github.com/simchowitzlabpublic/nano-world-model.git
cd nano-world-model
conda env create -f environment.yml && conda activate nanowm
```

Set data + results paths (or use the gitignored `src/configs/local/paths.yaml` template — see [docs/config_system.md](docs/config_system.md#path-configuration)):

```bash
export DATASET_DIR=/path/to/dino_wm_data       # DINO-WM envs (point_maze, pusht, ...)
export CSGO_DATA_DIR=/path/to/csgo             # CSGO HDF5 files
export RT1_DATA_ROOT=/path/to/rt1_fractal      # RT-1 LeRobot mirror (optional)
export RESULTS_DIR=/path/to/results            # checkpoints + logs land here
```

Download the i3d torchscript used by FID/FVD evaluation:

```bash
mkdir -p pretrained_models/i3d && curl -L \
    "https://www.dropbox.com/scl/fi/c5nfs6c422nlpj880jbmh/i3d_torchscript.pt?rlkey=x5xcjsrz0818i4qxyoglp5bb8&dl=1" \
    -o pretrained_models/i3d/i3d_torchscript.pt
```

For dataset downloads (DINO-WM, RT-1, CSGO, BlockWorld), see [docs/datasets/README.md](docs/datasets/README.md).

## 🥷 Train your first model

DINO-WM PushT, NanoWM-B/2, default settings (pred-v · additive injection · cosine + ZTSNR):

```bash
python src/main.py experiment=dino_wm_pusht dataset=dino_wm/pusht model=nanowm_b2
```

CSGO with the L/2 model:

```bash
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo
```

RT-1 (fractal) main run:

```bash
python src/main.py experiment=rt1 dataset=rt1/rt1 model=nanowm_b2
```

See [docs/training.md](docs/training.md) for the full training guide, design choices, and ablation tables.

## 📦 Pretrained Checkpoints

Best-config runs (pred-v · additive · cosine + ZTSNR · NanoWM-B/2 unless noted):

<div align="center">

| Domain | Checkpoint | Steps |
|:-------|:-----------|:------|
| DINO-WM Point Maze | 🤗 [nanowm-b2-dino-wm-point-maze-30k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-point-maze-30k) | 30k |
| DINO-WM Wall | 🤗 [nanowm-b2-dino-wm-wall-15k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-wall-15k) | 15k |
| DINO-WM Rope | 🤗 [nanowm-b2-dino-wm-rope-15k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-rope-15k) | 15k |
| DINO-WM Granular | 🤗 [nanowm-b2-dino-wm-granular-15k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-granular-15k) | 15k |
| DINO-WM PushT | 🤗 [nanowm-b2-dino-wm-pusht-100k](https://huggingface.co/knightnemo/nanowm-b2-dino-wm-pusht-100k) | 100k |
| RT-1 (fractal) | 🤗 [nanowm-b2-rt1-300k](https://huggingface.co/knightnemo/nanowm-b2-rt1-300k) | 300k |
| CSGO | 🤗 [nanowm-l2-csgo-100k](https://huggingface.co/knightnemo/nanowm-l2-csgo-100k) (NanoWM-L/2) | 100k |

</div>

We also ship 11 RT-1 ablation arms (one HF checkpoint per axis × method). See [docs/training.md#design-choices](docs/training.md#design-choices) for the full table and ablation numbers.

## 🎬 Sample Predictions

CSGO 50-frame auto-regressive long-rollouts (NanoWM-L/2, 100k):

<div align="center">

![CSGO 50-frame autoregressive long rollout](assets/csgo_100k_long_rollout.gif)

</div>

Quantitative Metrics
Evaluated on 256 fixed samples (seed=42), 250 DDIM steps, sequential scheduling (frame-by-frame autoregressive denoising).

<div align="center">

| Dataset | Steps | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|:--------|:------|:-------|:-------|:--------|:------|
| Point Maze | 30k | 36.74 | 0.984 | 0.019 | 9.66 |
| Wall | 15k | 34.05 | 0.994 | 0.010 | 2.64 |
| PushT | 100k | 33.19 | 0.982 | 0.016 | 13.63 |
| Rope | 15k | 31.63 | 0.953 | 0.056 | 35.20 |
| Granular | 15k | 26.08 | 0.917 | 0.073 | 40.05 |
| RT-1 | 300k | 24.36 | 0.787 | 0.180 | 35.08 |

</div>

Full per-domain numbers and methodology in [docs/evaluation.md](docs/evaluation.md).

## 🧭 Applications

NanoWM rollouts can be used directly for downstream applications, including long-horizon generation, video-to-3D reconstruction, and MPC-style planning.

<div align="center">

![Video-to-3D point cloud demo](assets/video_to_3d.gif)

</div>

- **[Long-horizon rollout](docs/applications/long_rollout.md)** — autoregressive rollout from trained checkpoints
- **[Video → 3D map](docs/applications/video_to_3d.md)** — Depth Anything 3 point cloud reconstruction from rollout videos
- **[MPC-style planning](docs/applications/planning.md)** — CEM planning over world model rollouts

## 📚 Documentation

- **[docs/config_system.md](docs/config_system.md)** — Hydra config layout, overrides, environment variables
- **[docs/training.md](docs/training.md)** — training workflow, design choices, ablation tables, all checkpoints
- **[docs/evaluation.md](docs/evaluation.md)** — evaluation workflow, metric definitions, full result tables
- **[docs/datasets/README.md](docs/datasets/README.md)** — DINO-WM / RT-1 / CSGO / BlockWorld formats, downloads, splits
- **[docs/applications/planning.md](docs/applications/planning.md)** — MPC + CEM model-predictive control
- **[docs/applications/long_rollout.md](docs/applications/long_rollout.md)** — long-horizon autoregressive rollout
- **[docs/applications/video_to_3d.md](docs/applications/video_to_3d.md)** — Depth Anything 3 point cloud pipeline
- **[src/wm_datasets/README.md](src/wm_datasets/README.md)** — DataSource API reference

## 🙏 Acknowledgements

We build upon a number of existing codebases: [Latte](https://github.com/Vchitect/Latte), [Vid2World](https://github.com/thuml/Vid2World), [DFoT](https://github.com/kwsong0113/diffusion-forcing-transformer), and [DINO-WM](https://github.com/gaoyuezhou/dino_wm). More broadly, this repository draws inspirations and design principles from [NanoGPT](https://github.com/karpathy/nanoGPT), [NanoChat](https://github.com/karpathy/nanochat), and [Boyuan Chen's Research Template](https://github.com/buoyancy99/research-template). We sincerely thank the codebases above for open-sourcing their works.

## 📝 Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@misc{nanoworldmodels,
  title={Nano World Model: A Minimalist, Batteries-Included Repository for Advancing World Model Science},
  author={Siqiao Huang and Partha Kaushik and Michael Chen and Hengkai Pan and Kaiwen Geng and Omar Chehab and Fernando Moreno-Pino and Max Simchowitz},
  year={2026},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/simchowitzlabpublic/nano-world-model}},
}
```
