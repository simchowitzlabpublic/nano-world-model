# Video to 3D Point Cloud

Generate colored 3D point clouds from MP4 videos using [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) multi-view depth estimation.

```
MP4 video → extract frames → restore aspect ratio (if rollout) → DA3 multi-view inference
                                                                → PLY point cloud
                                                                → viser 3D viewer (optional)
                                                                → depth visualizations (optional)
```

DA3 jointly estimates per-frame depth maps and camera parameters from multiple frames, enabling consistent 3D reconstruction. The pipeline unprojects depth maps to world-space 3D points, filters by confidence, and saves the result as a PLY file.

<div align="center">

![Video-to-3D point cloud demo](../../assets/video_to_3d.gif)

</div>

## Setup

The base training stack plus DA3 + viser:

```bash
conda env create -f environment.yml && conda activate nanowm   # base stack

# Video-to-3D extras
pip install viser xformers
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3 && pip install -e . && cd ..
```

Verify:
```bash
python -c "from depth_anything_3.api import DepthAnything3; import viser; print('OK')"
```

## Usage

### Basic

```bash
python src/scripts/video_to_pointcloud.py \
    --video sample_videos/train_sample_0.mp4 \
    --output output/scene.ply
```

### From a long-rollout video (restore native aspect ratio)

Long-rollout videos are stretched to the model's training resolution. For accurate depth, restore native aspect with `--native_res H W`:

```bash
# CSGO rollout: native 150×280
python src/scripts/video_to_pointcloud.py \
    --video results/long_rollout/csgo_100k/sample_0000_gen.mp4 \
    --output output/csgo_scene.ply \
    --native_res 150 280 \
    --visualize
```

If your video is already at the correct aspect ratio, omit `--native_res`. PixelWM rollout MP4s follow the same RGB input contract; see the [PixelWM guide](pixelwm.md).

### With interactive viewer

```bash
python src/scripts/video_to_pointcloud.py \
    --video sample_videos/train_sample_0.mp4 \
    --output output/scene.ply \
    --visualize
```

Opens a browser-based viser viewer with:
- Per-frame colored point clouds, timeline playback, and accumulation
- Camera frustums with RGB thumbnails (current frame highlighted)
- GUI sidebar with RGB and depth map panels
- Controls: point size, camera scale, playback FPS

### With depth visualizations

```bash
python src/scripts/video_to_pointcloud.py \
    --video sample_videos/train_sample_0.mp4 \
    --output output/scene.ply \
    --save_depth_vis
```

Per-frame PNGs (original | depth | confidence) under `output/scene_depth_vis/`.

### View existing results offline (no GPU)

```bash
# Rich viewer (timeline + depth + frustums)
python src/scripts/video_to_pointcloud.py --view output/scene_viewer.npz

# Simple viewer (point cloud only)
python src/scripts/video_to_pointcloud.py --view output/scene.ply
```

## CLI reference

<div align="center">

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--video` | required | Input MP4 |
| `--output` | required | Output PLY path |
| `--model` | `depth-anything/DA3-LARGE-1.1` | DA3 HF model ID |
| `--max_frames` | 30 | Cap on frames extracted |
| `--frame_step` | 1 | Extract every Nth frame |
| `--native_res H W` | None | Restore aspect ratio (e.g. `150 280` for CSGO) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--conf_threshold` | 40.0 | Confidence percentile for point filtering (0–100) |
| `--max_points` | 1000000 | Cap on output points |
| `--process_res` | 504 | DA3 internal processing resolution |
| `--save_depth_vis` | flag | Save per-frame depth PNGs |
| `--visualize` | flag | Launch viser viewer after inference |
| `--view <path>` | None | View existing `.npz` (rich) or `.ply` (simple) without re-running DA3 |

</div>

## DA3 model variants

<div align="center">

| Model ID | Notes |
|:---------|:------|
| `depth-anything/DA3-SMALL` | smallest, fastest |
| `depth-anything/DA3-BASE` | balance |
| `depth-anything/DA3-LARGE-1.1` | **default**, recommended |
| `depth-anything/DA3-GIANT-1.1` | most accurate |
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 1.40B, best quality |

</div>

## Output structure

```
output/
├── scene.ply                # colored point cloud
├── scene_viewer.npz         # state for offline rich viewer (--view)
├── scene_da3_export/        # DA3 native export (depth, confidence, cameras)
│   └── *.npz
└── scene_depth_vis/         # depth PNGs (with --save_depth_vis)
    ├── frame_0000.png
    └── ...
```

## Tips

- **Rollout videos**: always pass `--native_res H W` for stretched rollouts. Aspect-ratio mismatch breaks DA3's depth estimation.
- **Frame selection**: for videos with little camera motion, raise `--frame_step` (e.g. 3) to get more diverse viewpoints.
- **Memory**: drop `--max_frames` (e.g. 15) on smaller GPUs; it's the dominant memory factor.
- **Quality**: lower `--conf_threshold` (e.g. 20) to keep more points; raise it (e.g. 60) for cleaner clouds.
- **Resolution**: raise `--process_res` (e.g. 756) for sharper depth at the cost of more memory.

## See also

- [long_rollout.md](long_rollout.md) — generating videos from a trained world model
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — upstream model
- [viser](https://github.com/nerfstudio-project/viser) — 3D web viewer
