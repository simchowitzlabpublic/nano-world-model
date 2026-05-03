# World Model Datasets

Unified data loading pipeline for training video world models with action-conditioned prediction.

## Architecture

```
Training Code (train_pl.py)
            ↓
    WorldModelDataset ← Slicing, normalization, batching
            ↓
        DataSource ← Load trajectories from storage
            ↓
    Files / HuggingFace ← Raw data
```

**Two-layer design:**
- **DataSource**: Loads trajectory data from files or APIs
- **WorldModelDataset**: Slices trajectories into training clips

**Key features:**
- Memory efficient: States/actions in RAM, frames loaded on-demand
- Unified interface: All datasets return `{video, action, video_name}`
- Flexible sampling: Exhaustive (pre-compute all slices) or random (sample at runtime)

## Quick Start

### Basic Usage

```python
from wm_datasets import create_world_model_dataset

# Create dataset
dataset = create_world_model_dataset(
    dataset_name="point_maze",
    data_path="/data/point_maze",
    num_frames=16,
    frame_interval=1,
    split="train",
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    video = batch["video"]    # [B, T, C, H, W]
    action = batch["action"]  # [B, T, action_dim]
```

### Create Train/Val Datasets

```python
from wm_datasets import create_train_val_datasets

train_dataset, val_dataset = create_train_val_datasets(
    dataset_name="point_maze",
    data_path="/data/point_maze",
    num_frames=16,
    frame_interval=3,
    split_ratio=0.9,
    slice_mode="exhaustive",  # or "random"
    stride=1,  # For exhaustive mode
)
```

## Supported Datasets

### DINO-WM Datasets

<div align="center">

| Dataset | Trajectories | State Dim | Action Dim | Avg Length |
|:--------|:-------------|:----------|:-----------|:-----------|
| **point_maze** | 2,000 | 4 | 2 | 100 |
| **wall** | 1,920 | 2 | 2 | 50 |
| **pusht** | 18,685 | 5 | 2 | ~109 |
| **rope** | TBD | TBD | TBD | TBD |
| **granular** | TBD | TBD | TBD | TBD |
| **blockworld** | config-dependent | 0 | 5 | 70–140 |

</div>

**Dataset structure:**
```
dino_wm_data/
├── point_maze/
│   ├── states.pth
│   ├── actions.pth
│   ├── seq_lengths.pth (optional)
│   └── obses/  # Video frames
├── pusht_noise/train/
│   ├── states.pth
│   ├── rel_actions.pth  # or abs_actions.pth
│   ├── seq_lengths.pth
│   └── obses/
└── ...
```

## API Reference

### `create_world_model_dataset()`

```python
def create_world_model_dataset(
    dataset_name: str,
    data_path: str = None,
    repo_id: str = None,
    num_frames: int = 16,
    frame_interval: int = 1,
    image_size: tuple = (256, 256),
    split: str = "train",
    split_ratio: float = 0.9,
    normalize_action: bool = True,
    slice_mode: str = "exhaustive",
    stride: int = 1,
    **kwargs
) -> WorldModelDataset
```

**Key parameters:**
- `dataset_name`: Dataset identifier (e.g., "point_maze", "pusht")
- `data_path`: Path to dataset directory (for filesystem datasets)
- `repo_id`: HuggingFace repo ID (for LeRobot datasets)
- `num_frames`: Number of frames per clip
- `frame_interval`: Frame skip (1=every frame, 2=every other frame)
- `slice_mode`: "exhaustive" (pre-compute slices) or "random" (sample at runtime)
- `stride`: Sliding window stride for exhaustive mode (1=maximum data utilization)

**Dataset-specific kwargs:**
- `use_relative_actions` (pusht): Use relative vs absolute actions
- `action_scale` (pusht): Action scaling factor
- `object_name` (deformable): "rope" or "granular"
- `action_dim` (blockworld): Number of discrete actions for one-hot encoding

### Sampling Modes

**Exhaustive Mode** (default):
```python
dataset = create_world_model_dataset(
    ...,
    slice_mode="exhaustive",
    stride=1,  # Extract every possible slice
)
# Result: From 100-frame trajectory → ~85 slices (stride=1)
```

**Random Mode**:
```python
dataset = create_world_model_dataset(
    ...,
    slice_mode="random",
)
# Result: Sample random clips at runtime (memory efficient)
```

## Configuration Files

All dataset parameters should be explicit in config files:

```yaml
# configs/dino_wm/pusht_train.yaml
dataset: "pusht"
data_path: "/data/pusht_noise/train"

# Dataset-specific parameters
use_relative_actions: true
action_scale: 100.0
with_velocity: true

# Training parameters
num_frames: 16
frame_interval: 1
slice_mode: "exhaustive"
stride: 1
normalize_action: true
```

## Testing

Test datasets with:

```bash
# Test DataSource layer
python src/wm_datasets/test_data_source.py --dataset point_maze

# Test WorldModelDataset
python src/wm_datasets/test_world_model_dataset.py
```

## Adding New Datasets

### Step 1: Implement DataSource

```python
# In src/wm_datasets/data_source.py

class MyDataSource(DataSource):
    def __init__(self, data_path: str, **kwargs):
        # Load data...
        pass

    def get_num_trajectories(self) -> int:
        return len(self.trajectories)

    def load_trajectory(self, index: int) -> TrajectoryData:
        # Return TrajectoryData(states, actions, seq_length, meta)
        pass

    def load_visual_frames(self, index: int, start: int, end: int, step: int) -> torch.Tensor:
        # Load and return frames
        pass
```

### Step 2: Register in Factory

```python
# In create_data_source()
if dataset_name == "my_dataset":
    return MyDataSource(data_path, **kwargs)
```

### Step 3: Create Config

```yaml
# configs/my_dataset/train.yaml
dataset: "my_dataset"
data_path: "/path/to/data"
num_frames: 16
# ... other parameters
```

## Troubleshooting

**"FileNotFoundError: states.pth not found"**
- Verify dataset path and structure
- Ensure all required files exist

**"Missing required config: use_relative_actions"**
- Add explicit parameter to config file
- Never rely on implicit defaults

**"Out of memory"**
- Reduce `num_frames` or batch size
- Use `slice_mode="random"` for memory efficiency
- Use `n_rollout` parameter to load subset of data

## Additional Resources

- **Architecture details**: See [CLAUDE.md](../CLAUDE.md)
- **Testing procedures**: See [dev.md](../dev.md)
- **Example configs**: See [configs/](../configs/)

---

**Last Updated**: 2026-02-03
