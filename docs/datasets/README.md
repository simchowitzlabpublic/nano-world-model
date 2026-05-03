# Datasets

The repo ships configs for four dataset families: **DINO-WM** (5 simulated environments), **RT-1 fractal** (real-robot LeRobot), **CSGO** (Counter-Strike deathmatch), and **BlockWorld** (FlowM 3D memory benchmark). All datasets feed through a common `WorldModelDataset` interface — see `src/wm_datasets/`.

<div align="center">

![Dataset overview](../../assets/dataset_overview.png)

</div>

## Setup paths

```bash
export DATASET_DIR=/path/to/dino_wm_data       # DINO-WM root
export CSGO_DATA_DIR=/path/to/csgo             # CSGO root (HDF5 files)
export RT1_DATA_ROOT=/path/to/rt1_fractal      # RT-1 (LeRobot mirror)
```

For BlockWorld, set `DATASET_DIR` to the parent directory containing `blockworld/`.

Or use the gitignored `src/configs/local/paths.yaml` (template at `paths.yaml.example`). See [config_system.md](../config_system.md#path-configuration).

## At a glance

<div align="center">

| Dataset | Episodes | Frames/ep | Resolution | Action dim | Train sampling | Notes |
|:--------|:---------|:----------|:-----------|:-----------|:---------------|:------|
| DINO-WM Point Maze | ~500 | ~100–200 | 256² | 2 | exhaustive | frame_interval=5 |
| DINO-WM PushT | ~1000 | ~100–200 | 256² | 2 | exhaustive | relative actions, action_scale=100 |
| DINO-WM Wall | ~500 | ~100–200 | 256² | 2 | exhaustive | |
| DINO-WM Rope | ~500 | ~100–200 | 256² | 2 | exhaustive | deformable |
| DINO-WM Granular | ~500 | ~100–200 | 256² | 2 | exhaustive | deformable |
| RT-1 (fractal) | 87k | ~40–60 | 256² | 7 | random | LeRobot v2.0, frame_interval=1 |
| CSGO | 5500 | 1000 | 320×512 | 51 | random | fixed val start indices, frame_interval=1 |
| BlockWorld | config-dependent | 70–140 | 128² | 5 | random | FlowM 3D memory benchmark |

</div>

---

## DINO-WM datasets

5 simulated environments (point_maze, pusht, wall, rope, granular). Originally from the [DINO-WM repo](https://github.com/gaoyuezhou/dino_wm). Pure vision (no state); 2D action.

### Download

From [OSF](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28). Unzip the relevant subfolders following the [DINO-WM README](https://github.com/gaoyuezhou/dino_wm). Layout we expect:

```
${DATASET_DIR}/
├── point_maze/
├── pusht_noise/
│   ├── train/
│   └── val/
├── wall_single/
└── deformable/   # rope + granular share this directory
```

### Configs

`src/configs/dataset/dino_wm/`:
- `base.yaml` — sets exhaustive train/val sampling, validation_size=32, action normalization on
- per-env: `point_maze.yaml`, `pusht.yaml`, `wall.yaml`, `rope.yaml`, `granular.yaml`

PushT is the most distinctive: it uses **relative actions** (delta moves), `action_scale=100`, and `with_velocity=True`.

### Train command

```bash
python src/main.py experiment=dino_wm_pusht dataset=dino_wm/pusht model=nanowm_b2
```

Swap the experiment + dataset names for the other four envs.

---

## RT-1 (fractal)

Real-robot manipulation, 87k episodes from the [IPEC-COMMUNITY/fractal20220817_data_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot) mirror of RT-1 fractal on HuggingFace. LeRobot v2.0 format. 7-D end-effector action.

### Download

```bash
huggingface-cli download IPEC-COMMUNITY/fractal20220817_data_lerobot \
    --repo-type dataset --local-dir $RT1_DATA_ROOT
```

(Run `huggingface-cli login` first if you haven't.) Without `RT1_DATA_ROOT` set, LeRobot will pull on first use into `~/.cache/huggingface/lerobot`.

### Config

`src/configs/dataset/rt1/rt1.yaml`:
- `data_path: "IPEC-COMMUNITY/fractal20220817_data_lerobot"` (HF repo id)
- `root: "${oc.env:RT1_DATA_ROOT,./data/rt1_fractal}"` (local mirror)
- `image_key: "observation.images.image"` (RT-1's frame field)
- `train_slice_mode: "random"` (87k episodes is too many for exhaustive sampling)
- `action_dim: 7`

### Train command

```bash
# Main run: NanoWM-B/2, 300k steps
python src/main.py experiment=rt1 dataset=rt1/rt1 model=nanowm_b2

# Ablation arms (50k steps)
python src/main.py experiment=ablation_rt1 dataset=rt1/rt1 model=nanowm_b2 \
    experiment.diffusion.pred_name=v   # or pred_name=x / epsilon
```

See [training.md](../training.md#design-choices) for the full ablation table.

---

## CSGO

Counter-Strike: Global Offensive deathmatch gameplay. Source: [teapearce/counter-strike_deathmatch](https://huggingface.co/datasets/teapearce/counter-strike_deathmatch). 5500 episodes (5000 train / 500 val), 1000 frames each at 3 FPS — ~675 GB total.

### Download

```bash
huggingface-cli download teapearce/counter-strike_deathmatch \
    --repo-type dataset --local-dir $CSGO_DATA_DIR
```

We use **only** the standard deathmatch files: `hdf5_dm_july2021_*.hdf5`. The 75 expert files (aim training, inferno/mirage/nuke expert) are excluded for consistency with the [Vid2World](https://github.com/thuml/Vid2World) evaluation protocol.

### Train / val split

The split is fixed (matches Vid2World):
- `src/wm_datasets/data_source/game/csgo_splits/train_split.txt` — 5000 files
- `src/wm_datasets/data_source/game/csgo_splits/test_split.txt` — 500 files
- `src/wm_datasets/data_source/game/csgo_splits/csgo_validation_start_indices.npy` — 500 fixed start frames (one per val episode)

The fixed start indices ensure identical eval positions across runs.

### Data format

Each `.hdf5` file is one episode:
```
frame_0_x: [150, 280, 3] uint8     # frame at timestep 0
frame_0_y: [51] float32            # action vector at timestep 0
...
frame_999_x: [150, 280, 3] uint8
frame_999_y: [51] float32
```

Frames: 150×280 native, resized to 320×512 during loading (`resize_mode: stretch`). After loading: `[0, 255] uint8 → [-1, 1] float32`. **Native aspect ratio is 1.87:1** — when running long rollout decoded back to native res, restore aspect with `--native_res 150 280` (see [applications/long_rollout.md](../applications/long_rollout.md)).

**Action space (51-D)**:

<div align="center">

| Dims | Description |
|:-----|:------------|
| 0–10 (11) | Keyboard keys (W/A/S/D, Space, Ctrl, Shift, …) — binary |
| 11–12 (2) | Mouse clicks (LMB, RMB) — binary |
| 13–35 (23) | Mouse X movement — one-hot bins |
| 36–50 (15) | Mouse Y movement — one-hot bins |

</div>

State dim: 0 (pure vision; no health/ammo/inventory).

### Why CSGO uses random sampling

5000 episodes × 1000 frames ≈ 5M slices. Exhaustive enumeration is impractical, so:
- `train_slice_mode: random` — sample a random `[start, start+T]` window per episode each iteration
- `val_slice_mode: exhaustive` — but only at the 500 fixed `val_start_indices`

### Config

`src/configs/dataset/game/csgo.yaml`:

```yaml
name: "csgo"
frame_interval: 1
loader:
  data_path: "${csgo_data_dir}"
  normalize_action: False     # actions are already [0, 1]
  resize_mode: "stretch"
  train_file_list: "src/wm_datasets/data_source/game/csgo_splits/train_split.txt"
  val_file_list:   "src/wm_datasets/data_source/game/csgo_splits/test_split.txt"
  val_start_indices: "src/wm_datasets/data_source/game/csgo_splits/csgo_validation_start_indices.npy"
  train_slice_mode: "random"
  val_slice_mode: "exhaustive"
spec:
  action_dim: 51
```

### Train command

```bash
python src/main.py experiment=csgo dataset=game/csgo model=nanowm_l2_csgo
```

The shipped checkpoints are NanoWM-L/2 trained for 50k or 100k steps. See [training.md](../training.md#pretrained-checkpoints-best-config-runs) for HF links.

### Memory

- Actions: ~10 MB/episode (cached in memory)
- Frames: ~123 MB/episode (loaded on demand — never cached)
- Total: ~675 GB on disk; RAM stays bounded by the per-batch decode

---

## BlockWorld

FlowM's 3D Dynamic BlockWorld memory benchmark. Source: [hlillemark/flowm](https://github.com/hlillemark/flowm). The dataset uses RGB videos plus per-frame discrete actions; nano-world-model exposes those actions as 5-D one-hot vectors.

### Download

Use the FlowM dataset downloader and place the extracted dataset under `${DATASET_DIR}/blockworld`:

```bash
cd /path/to/flowm
bash ./download_datasets.sh --dataset blockworld --configs dynamic --splits train,validation
```

### Data format

Expected layout:

```text
${DATASET_DIR}/blockworld/
├── sunday_v2_training/
│   └── 0/
│       ├── 0000_rgb.mp4
│       ├── 0000_depth.mp4
│       └── 0000_actions.pt
└── sunday_v2_validation/
```

The `*_actions.pt` sidecar must contain either an `actions` tensor of integer action ids or an already-vectorized `[T, 5]` tensor. Depth videos are left untouched; the current integration is RGB-only to match the existing NanoWM video/action interface.

### Config

`src/configs/dataset/memory/blockworld.yaml`:

```yaml
name: "blockworld"
frame_interval: 1
loader:
  data_path_train: "${dataset_dir}/blockworld/sunday_v2_training"
  data_path_val: "${dataset_dir}/blockworld/sunday_v2_validation"
  normalize_action: False
  train_slice_mode: "random"
  val_slice_mode: "exhaustive"
spec:
  action_dim: 5
```

### Train command

```bash
python src/main.py dataset=memory/blockworld model=nanowm_b2
```

### Memory

- Actions are small and cached on first trajectory access.
- Frames are decoded on demand from MP4 and are never cached by the data source.

---

## Adding a new dataset

1. **DataSource** in `src/wm_datasets/data_source/`:
   ```python
   class MyDataSource(DataSource):
       def load_trajectory(self, index): ...
       def load_visual_frames(self, index, start, end, step=1): ...
       def get_num_trajectories(self): ...
       def get_seq_length(self, index): ...
       @property
       def action_dim(self): ...
   ```

2. **Register** in `src/wm_datasets/data_source/factory.py` (add a branch on `dataset_name`).

3. **Config** under `src/configs/dataset/<family>/<name>.yaml` inheriting from a `base.yaml`:
   ```yaml
   defaults:
     - <family>/base
   name: "my_dataset"
   loader:
     data_path: "${dataset_dir}/my_dataset"
     train_slice_mode: "exhaustive"   # or "random"
   spec:
     action_dim: <int>
   ```

4. **Experiment config** (optional) under `src/configs/experiment/<name>.yaml` if you want non-default training knobs.

Run with `dataset=<family>/<name>`.

## See also

- [config_system.md](../config_system.md) — full Hydra reference
- [training.md](../training.md) — training workflow + design choices
- [evaluation.md](../evaluation.md) — eval workflow + result tables
- [`src/wm_datasets/README.md`](../../src/wm_datasets/README.md) — DataSource API reference
