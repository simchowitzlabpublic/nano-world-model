"""
WorldModelDataset - Unified dataset for world model training.

This module provides a PyTorch Dataset that wraps DataSource and provides:
- Sliding window slicing from long trajectories
- Train/validation split
- Data normalization (actions, states, pixels)
- Integration with PyTorch DataLoader
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from omegaconf import OmegaConf, ListConfig
from torch.utils.data import Dataset
from dataclasses import dataclass

from . import stats_cache
from .data_source import DataSource, create_data_source


@dataclass
class SliceInfo:
    """Information about a sliced video clip."""
    traj_idx: int           # Source trajectory index
    start_frame: int        # Start frame in trajectory
    end_frame: int          # End frame in trajectory (exclusive)
    actual_length: int      # Actual number of frames (≤ num_frames)


class WorldModelDataset(Dataset):
    """
    Unified dataset for world model training.

    Wraps a DataSource and provides sliced video clips with actions for training.
    Handles normalization, train/val split, and efficient data loading.

    Args:
        data_source: DataSource instance (DinoWorldModelDataSource or LeRobotDataSource)
        num_frames: Number of frames per clip
        frame_interval: Frame skip (1 = every frame, 2 = every other frame)
        image_size: Target image size (H, W)
        split: "train" or "val"
        split_ratio: Ratio of training data (0-1)
        normalize_action: Whether to normalize actions to [-1, 1]
        normalize_state: Whether to normalize states to [-1, 1]
        normalize_pixel: Whether to normalize pixels to [-1, 1]
        random_seed: Random seed for train/val split
        slice_mode: Slicing strategy ("exhaustive" or "random")
            - "exhaustive": Pre-compute all possible slices (stride=1), deterministic
            - "random": Sample random clips at runtime, more memory efficient
        stride: Stride for exhaustive mode (default: 1 for maximum data utilization)
                Ignored in random mode
        video_augmentation: Optional video augmentation for clips
    """

    def __init__(
        self,
        data_source: DataSource,
        num_frames: int = 16,
        frame_interval: int = 1,
        image_size: Tuple[int, int] = (256, 256),
        split: str = "train",
        split_ratio: float = 0.9,
        normalize_action: bool = True,
        normalize_state: bool = False,
        normalize_pixel: bool = True,
        random_seed: int = 42,
        slice_mode: str = "exhaustive",
        stride: int = 1,
        video_augmentation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fixed_start_indices: Optional[np.ndarray] = None,
        resize_mode: str = "stretch",
        use_data_source_stats: bool = True,
        _inherit_stats_from: Optional["WorldModelDataset"] = None,
        precomputed_slices: Optional[List[Dict[str, int]]] = None,
    ):
        super().__init__()

        # Validate slice_mode
        if slice_mode not in ["exhaustive", "random"]:
            raise ValueError(f"slice_mode must be 'exhaustive' or 'random', got {slice_mode}")

        # Validate resize_mode
        if resize_mode not in ["stretch", "pad"]:
            raise ValueError(f"resize_mode must be 'stretch' or 'pad', got {resize_mode}")

        self.data_source = data_source
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        # Ensure image_size is a tuple of ints (Hydra may pass ListConfig)
        if isinstance(image_size, ListConfig):
            image_size = OmegaConf.to_container(image_size, resolve=True)
        if isinstance(image_size, (list, tuple)):
            if len(image_size) == 1:
                self.image_size = (int(image_size[0]), int(image_size[0]))
            elif len(image_size) == 2:
                self.image_size = (int(image_size[0]), int(image_size[1]))
            else:
                raise ValueError(f"image_size must have length 1 or 2, got {len(image_size)}")
        else:
            self.image_size = (int(image_size), int(image_size))
        self.split = split
        self.split_ratio = split_ratio
        self.normalize_action = normalize_action
        self.normalize_state = normalize_state
        self.normalize_pixel = normalize_pixel
        self.random_seed = random_seed
        self.slice_mode = slice_mode
        self.stride = stride
        self.resize_mode = resize_mode
        self.video_augmentation = video_augmentation
        self.fixed_start_indices = fixed_start_indices
        self.use_data_source_stats = use_data_source_stats
        self._inherit_stats_from = _inherit_stats_from

        # Get dataset info
        self.action_dim = data_source.action_dim * self.frame_interval
        self.state_dim = data_source.state_dim
        self.num_trajectories = data_source.get_num_trajectories()

        # Setup random generator for random mode
        self.rng = np.random.RandomState(random_seed)

        if precomputed_slices is not None:
            # Fast path: skip exhaustive slice creation, use pre-built specs directly
            self.all_slices = [
                SliceInfo(
                    traj_idx=s["traj_idx"],
                    start_frame=s["start_frame"],
                    end_frame=s["end_frame"],
                    actual_length=self.num_frames,
                )
                for s in precomputed_slices
            ]
            self.slice_indices = list(range(len(self.all_slices)))
            print(f"Split '{self.split}': {len(self.slice_indices)} slices (precomputed, skipped indexing)")
        elif self.slice_mode == "exhaustive":
            self._split_trajectories_indices()
            self._create_slices_exhaustive(self.traj_indices, fixed_start_indices)
            self.slice_indices = list(range(len(self.all_slices)))
            print(f"Split '{self.split}': {len(self.slice_indices)} slices")
        else:
            self._split_trajectories()

        if self._inherit_stats_from is not None:
            self._inherit_stats(self._inherit_stats_from)
        else:
            if self.normalize_action:
                self._compute_action_stats()
            if self.normalize_state:
                self._compute_state_stats()

    def _create_slices_exhaustive(self, traj_indices: Optional[List[int]] = None, fixed_start_indices: Optional[np.ndarray] = None):
        """
        Create all possible slices from trajectories using sliding window (exhaustive mode).

        For each trajectory, we create overlapping clips of length num_frames.
        The stride determines how much we move the window each time (default: 1 for maximum coverage).

        If fixed_start_indices is provided, only create one slice per trajectory at the specified start position.
        This is used for validation to ensure consistent evaluation (e.g., Vid2World CSGO evaluation).
        """
        self.all_slices: List[SliceInfo] = []

        if traj_indices is None:
            traj_indices = list(range(self.num_trajectories))

        # Check if using fixed start indices (for consistent validation)
        if fixed_start_indices is not None:
            if len(fixed_start_indices) != len(traj_indices):
                raise ValueError(
                    f"fixed_start_indices length ({len(fixed_start_indices)}) must match "
                    f"number of trajectories ({len(traj_indices)})"
                )
            print(f"Using fixed start indices for validation (Vid2World compatible)")

            skipped_trajs = 0
            for i, traj_idx in enumerate(traj_indices):
                seq_length = self._get_seq_length(traj_idx)

                # Calculate required frames accounting for frame_interval
                required_frames = self.num_frames * self.frame_interval
                start_frame = int(fixed_start_indices[i])
                end_frame = start_frame + required_frames

                if end_frame > seq_length:
                    print(f"Warning: Trajectory {traj_idx} start_frame {start_frame} + required {required_frames} > length {seq_length}, skipping")
                    skipped_trajs += 1
                    continue

                self.all_slices.append(SliceInfo(
                    traj_idx=traj_idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    actual_length=self.num_frames
                ))

            print(f"Created {len(self.all_slices)} slices from fixed start indices")
            if skipped_trajs > 0:
                print(f"Skipped {skipped_trajs} trajectories with invalid start positions")
            return

        # Original sliding window logic for training
        print(f"Creating slices (exhaustive mode, stride={self.stride}) from {len(traj_indices)} trajectories...")
        skipped_trajs = 0

        for traj_idx in traj_indices:
            seq_length = self._get_seq_length(traj_idx)

            # Calculate required frames accounting for frame_interval
            required_frames = self.num_frames * self.frame_interval

            if seq_length < required_frames:
                skipped_trajs += 1
                continue

            for start_frame in range(0, seq_length - required_frames + 1, self.stride):
                end_frame = start_frame + required_frames
                self.all_slices.append(SliceInfo(
                    traj_idx=traj_idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    actual_length=self.num_frames
                ))

        print(f"Created {len(self.all_slices)} slices total")
        if skipped_trajs > 0:
            print(f"Skipped {skipped_trajs} trajectories that were too short")

    def _split_trajectories_indices(self):
        np.random.seed(self.random_seed)

        # Shuffle trajectory indices
        all_traj_indices = np.arange(self.num_trajectories)
        np.random.shuffle(all_traj_indices)

        # Split by ratio
        split_idx = int(len(all_traj_indices) * self.split_ratio)

        if self.split == "train":
            self.traj_indices = all_traj_indices[:split_idx].tolist()
        elif self.split == "val":
            self.traj_indices = all_traj_indices[split_idx:].tolist()
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'")

    def _get_seq_length(self, traj_idx: int) -> int:
        """Get sequence length without forcing action load when possible."""
        if hasattr(self.data_source, "get_seq_length"):
            return self.data_source.get_seq_length(traj_idx)
        traj_data = self.data_source.load_trajectory(traj_idx)
        return traj_data.seq_length

    def _split_trajectories(self):
        self._split_trajectories_indices()

        # Pre-compute valid ranges for each trajectory (for efficiency)
        self.traj_valid_ranges = []
        skipped_trajs = 0
        for traj_idx in self.traj_indices:
            seq_length = self._get_seq_length(traj_idx)
            required_frames = self.num_frames * self.frame_interval

            if seq_length >= required_frames:
                max_start = seq_length - required_frames
                self.traj_valid_ranges.append((traj_idx, 0, max_start))
            else:
                # Too short, skip in random sampling mode
                skipped_trajs += 1

        print(f"Split '{self.split}': {len(self.traj_valid_ranges)} trajectories (random sampling mode)")
        if skipped_trajs > 0:
            print(f"Skipped {skipped_trajs} trajectories that were too short")

    def _split_train_val(self):
        """Split slices into train and validation sets."""
        np.random.seed(self.random_seed)

        # Shuffle slices
        indices = np.random.permutation(len(self.all_slices))

        # Split by ratio
        split_idx = int(len(indices) * self.split_ratio)

        if self.split == "train":
            self.slice_indices = indices[:split_idx].tolist()
        elif self.split == "val":
            self.slice_indices = indices[split_idx:].tolist()
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'")

        print(f"Split '{self.split}': {len(self.slice_indices)} slices")

    def get_slice_spec(self, slice_idx: int) -> Dict[str, int]:
        """
        Return a stable slice spec for saving/loading fixed subsets.
        """
        slice_info = self.all_slices[slice_idx]
        return {
            "traj_idx": slice_info.traj_idx,
            "start_frame": slice_info.start_frame,
            "end_frame": slice_info.end_frame,
        }

    def sample_fixed_slice_specs(self, size: int, seed: int = 42) -> List[Dict[str, int]]:
        """
        Sample a fixed set of slice specs from current split.
        """
        if self.slice_mode != "exhaustive":
            raise ValueError("Fixed subset requires slice_mode='exhaustive'")
        if size <= 0:
            raise ValueError("Fixed subset size must be > 0")
        rng = np.random.RandomState(seed)
        num_available = len(self.slice_indices)
        if size > num_available:
            size = num_available
        selected = rng.choice(num_available, size=size, replace=False)
        return [self.get_slice_spec(self.slice_indices[i]) for i in selected]

    def set_fixed_slices(self, slice_specs: List[Dict[str, int]]) -> None:
        """
        Set current split to a fixed list of slices.
        """
        if self.slice_mode != "exhaustive":
            raise ValueError("Fixed subset requires slice_mode='exhaustive'")
        index_map = {
            (s.traj_idx, s.start_frame, s.end_frame): i
            for i, s in enumerate(self.all_slices)
        }
        new_indices = []
        for spec in slice_specs:
            key = (spec["traj_idx"], spec["start_frame"], spec["end_frame"])
            if key not in index_map:
                raise ValueError(f"Slice not found in dataset: {spec}")
            new_indices.append(index_map[key])
        self.slice_indices = new_indices

    def _compute_action_stats(self):
        """Fill _raw_action_mean/std (disk cache → data source → compute)."""
        cache_path = self._cache_path("action")
        cached = stats_cache.load(cache_path)
        if cached is not None:
            self._raw_action_mean, self._raw_action_std = cached
            self._broadcast_action_stats()
            print(f"Loaded cached action stats ({self.split}) from disk")
            return

        if self.use_data_source_stats:
            mean, std = stats_cache.try_source_stats(self.data_source, "action")
            if mean is not None:
                self._raw_action_mean = mean
                self._raw_action_std = std + 1e-6
                self._broadcast_action_stats()
                stats_cache.save(cache_path, self._raw_action_mean, self._raw_action_std)
                print(
                    f"Using data source action stats for split '{self.split}': "
                    f"raw mean={self._raw_action_mean}, raw std={self._raw_action_std}"
                )
                return

        print("Computing action normalization statistics...")
        split_indices = self._split_indices()
        chunks = []
        for i in split_indices:
            traj = self.data_source.load_trajectory(i)
            chunks.append(traj.actions[: traj.seq_length])
        all_actions = torch.cat(chunks, dim=0)
        self._raw_action_mean = all_actions.mean(dim=0)
        self._raw_action_std = all_actions.std(dim=0) + 1e-6
        self._broadcast_action_stats()
        stats_cache.save(cache_path, self._raw_action_mean, self._raw_action_std)
        print(
            f"Action stats computed from {len(split_indices)} {self.split} trajectories: "
            f"raw mean={self._raw_action_mean}, raw std={self._raw_action_std}"
        )

    def _compute_state_stats(self):
        """Fill state_mean/std (disk cache → data source → compute)."""
        cache_path = self._cache_path("state")
        cached = stats_cache.load(cache_path)
        if cached is not None:
            self.state_mean, self.state_std = cached
            print(f"Loaded cached state stats ({self.split}) from disk")
            return

        if self.use_data_source_stats:
            mean, std = stats_cache.try_source_stats(self.data_source, "state")
            if mean is not None:
                self._check_state_std(std)
                self.state_mean = mean
                self.state_std = std + 1e-6
                stats_cache.save(cache_path, self.state_mean, self.state_std)
                print(f"Using data source state stats for split '{self.split}'")
                return

        print("Computing state normalization statistics...")
        split_indices = self._split_indices()
        chunks = []
        for i in split_indices:
            traj = self.data_source.load_trajectory(i)
            chunks.append(traj.states[: traj.seq_length])
        all_states = torch.cat(chunks, dim=0)
        raw_std = all_states.std(dim=0)
        self._check_state_std(raw_std)
        self.state_mean = all_states.mean(dim=0)
        self.state_std = raw_std + 1e-6
        stats_cache.save(cache_path, self.state_mean, self.state_std)
        print(f"State stats computed: mean={self.state_mean}, std={self.state_std}")

    def _broadcast_action_stats(self) -> None:
        """Tile raw stats to match self.action_dim (post frame_interval reshape)."""
        if self.frame_interval > 1:
            self.action_mean = self._raw_action_mean.repeat(self.frame_interval)
            self.action_std = self._raw_action_std.repeat(self.frame_interval)
        else:
            self.action_mean = self._raw_action_mean
            self.action_std = self._raw_action_std

    def _split_indices(self) -> List[int]:
        indices = getattr(self, "traj_indices", None)
        if indices is None:
            indices = list(range(self.num_trajectories))
        return indices

    def _cache_path(self, kind: str):
        dim = self.data_source.action_dim if kind == "action" else self.data_source.state_dim
        return stats_cache.cache_path(
            self.data_source, kind, self.split, self._split_indices(),
            self.num_trajectories, dim,
        )

    @staticmethod
    def _check_state_std(std: torch.Tensor) -> None:
        # Vision-only datasets (RT-1, CSGO) return placeholder states with std≈0;
        # dividing by that blows up the network. Fail loudly instead.
        if torch.any(std < 1e-4):
            raise ValueError(
                f"normalize_state=True but state std is effectively zero ({std}). "
                "Set normalize_state=False."
            )

    def _inherit_stats(self, other: "WorldModelDataset") -> None:
        """Copy stats from `other` (used so val reuses train's distribution)."""
        if self.normalize_action:
            self._raw_action_mean = other._raw_action_mean
            self._raw_action_std = other._raw_action_std
            self._broadcast_action_stats()
        if self.normalize_state:
            self.state_mean = other.state_mean
            self.state_std = other.state_std

    def __len__(self) -> int:
        """Return number of samples in this split."""
        if self.slice_mode == "exhaustive":
            return len(self.slice_indices)
        else:
            # In random mode, one epoch = one sample per trajectory (len = num_trajectories)
            return len(self.traj_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single video clip with actions.

        Args:
            idx: Index in the current split

        Returns:
            Dictionary containing:
                - "video": [T, C, H, W] tensor, normalized to [-1, 1] if normalize_pixel=True
                - "action": [T, action_dim] tensor, normalized if normalize_action=True
                - "state": [T, state_dim] tensor, normalized if normalize_state=True (optional)
                - "video_name": str, unique identifier for this clip
        """
        if self.slice_mode == "exhaustive":
            return self._getitem_exhaustive(idx)
        else:
            return self._getitem_random(idx)

    def _getitem_exhaustive(self, idx: int) -> Dict[str, Any]:
        """Get item in exhaustive mode (use pre-computed slices)."""
        # Get slice info
        slice_idx = self.slice_indices[idx]
        slice_info = self.all_slices[slice_idx]

        return self._load_slice(slice_info)

    def _getitem_random(self, idx: int) -> Dict[str, Any]:
        """Get item in random mode (sample random clip at runtime)."""
        # idx is just used to seed the random state for this call
        # In practice, we randomly select a trajectory and a start point

        # Select random trajectory
        traj_entry = self.traj_valid_ranges[idx % len(self.traj_valid_ranges)]
        traj_idx, min_start, max_start = traj_entry

        # Select random start point
        if max_start > min_start:
            start_frame = self.rng.randint(min_start, max_start + 1)
        else:
            start_frame = min_start

        # Calculate end frame
        required_frames = self.num_frames * self.frame_interval
        traj_data = self.data_source.load_trajectory(traj_idx)
        end_frame = min(start_frame + required_frames, traj_data.seq_length)

        # Create slice info on-the-fly
        slice_info = SliceInfo(
            traj_idx=traj_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            actual_length=min(self.num_frames, (end_frame - start_frame + self.frame_interval - 1) // self.frame_interval)
        )

        return self._load_slice(slice_info)

    def _load_slice(self, slice_info: SliceInfo) -> Dict[str, Any]:
        """
        Load a slice given SliceInfo.

        This is the shared loading logic for both modes.
        """

        # Load trajectory data (cached in DataSource)
        traj_data = self.data_source.load_trajectory(slice_info.traj_idx)

        actions_full = traj_data.actions[slice_info.start_frame:slice_info.end_frame]
        states = traj_data.states[slice_info.start_frame:slice_info.end_frame:self.frame_interval]
        states = states[:self.num_frames]

        # Load visual frames (on-demand from DataSource)
        # DataSource API: load_visual_frames(index, start, end, step)
        video = self.data_source.load_visual_frames(
            index=slice_info.traj_idx,
            start=slice_info.start_frame,
            end=slice_info.end_frame,
            step=self.frame_interval
        )  # [T, C, H, W]

        # Trim to exactly num_frames
        video = video[:self.num_frames]

        # Apply video augmentation if provided
        if self.video_augmentation is not None:
            if self.split == "val":
                print("[WARNING] Applying video augmentation to validation set, not recommended.")
            video = self.video_augmentation(video)

        # Resize video if needed
        if video.shape[-2:] != self.image_size:
            if self.resize_mode == "stretch":
                video = torch.nn.functional.interpolate(
                    video,
                    size=self.image_size,
                    mode='bilinear',
                    align_corners=False
                )
            elif self.resize_mode == "pad":
                # Letterbox resize: scale proportionally, pad with black
                T, C, H, W = video.shape
                target_h, target_w = self.image_size
                scale = min(target_h / H, target_w / W)
                new_h = int(H * scale)
                new_w = int(W * scale)
                video = torch.nn.functional.interpolate(
                    video, size=(new_h, new_w), mode='bilinear', align_corners=False
                )
                pad_h = target_h - new_h
                pad_w = target_w - new_w
                pad_top = pad_h // 2
                pad_left = pad_w // 2
                # F.pad format: (left, right, top, bottom)
                video = torch.nn.functional.pad(
                    video,
                    (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
                    value=0.0
                )

        if self.normalize_action:
            # Normalize on raw per-dim stats (shape matches actions_full's last dim).
            actions_full = (actions_full - self._raw_action_mean) / self._raw_action_std

        if self.frame_interval > 1:
            required_actions = self.num_frames * self.frame_interval
            actions_full = actions_full[:required_actions]
            actions = actions_full.reshape(self.num_frames, self.frame_interval, -1).reshape(self.num_frames, -1)
        else:
            actions = actions_full[::self.frame_interval]
            actions = actions[:self.num_frames]

        if self.normalize_state:
            states = (states - self.state_mean) / self.state_std

        # Normalize pixels to [-1, 1]
        if self.normalize_pixel:
            # Assume video is in [0, 1] range
            video = video * 2.0 - 1.0

        # Create unique video name
        video_name = f"traj_{slice_info.traj_idx:04d}_start_{slice_info.start_frame:04d}"

        # Prepare output
        output = {
            "video": video,          # [T, C, H, W]
            "action": actions,       # [T, action_dim]
            "video_name": video_name,
            "meta_info": {
                "traj_idx": slice_info.traj_idx,
                "start_idx": slice_info.start_frame,
            },
        }

        # Optionally include states
        if self.state_dim > 0:
            output["state"] = states  # [T, state_dim]

        # Include any metadata
        if traj_data.meta:
            output["metadata"] = traj_data.meta

        return output

    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get normalization statistics for inference/visualization.

        Returns:
            Dictionary with keys:
                - "action_mean", "action_std" (if normalize_action=True)
                - "state_mean", "state_std" (if normalize_state=True)
        """
        stats = {}

        if self.normalize_action:
            stats["action_mean"] = self.action_mean
            stats["action_std"] = self.action_std

        if self.normalize_state:
            stats["state_mean"] = self.state_mean
            stats["state_std"] = self.state_std

        return stats


def create_world_model_dataset(
    dataset_name: str,
    data_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    num_frames: int = 16,
    frame_interval: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    split: str = "train",
    split_ratio: float = 0.9,
    normalize_action: bool = True,
    normalize_state: bool = False,
    normalize_pixel: bool = True,
    n_rollout: Optional[int] = None,
    random_seed: int = 42,
    slice_mode: str = "exhaustive",
    stride: int = 1,
    fixed_start_indices: Optional[np.ndarray] = None,
    resize_mode: str = "stretch",
    **kwargs
) -> WorldModelDataset:
    """
    Factory function to create WorldModelDataset from config.

    Args:
        dataset_name: Name of dataset (e.g., "point_maze", "pusht", "rt1")
        data_path: Path to dataset directory (for filesystem datasets)
        repo_id: HuggingFace repo ID (for LeRobot datasets)
        num_frames: Number of frames per clip
        frame_interval: Frame skip (1 = every frame, 2 = every other frame)
        image_size: Target image size (H, W)
        split: "train" or "val"
        split_ratio: Ratio of training data
        normalize_action: Whether to normalize actions
        normalize_state: Whether to normalize states
        normalize_pixel: Whether to normalize pixels
        n_rollout: Number of trajectories to load (None = all)
        random_seed: Random seed for reproducibility
        slice_mode: "exhaustive" (pre-compute all slices) or "random" (sample at runtime)
        stride: Stride for exhaustive mode (default: 1 for maximum coverage)
        fixed_start_indices: Fixed start frame indices for validation (for Vid2World consistency)
        **kwargs: Additional dataset-specific parameters

    Returns:
        WorldModelDataset instance
    """
    # Create data source
    data_source_kwargs = {
        "dataset_name": dataset_name,
        "n_rollout": n_rollout,
    }

    # Add data_path or repo_id depending on dataset type
    if data_path is not None:
        data_source_kwargs["data_path"] = data_path
    if repo_id is not None:
        data_source_kwargs["repo_id"] = repo_id

    # Add dataset-specific kwargs that DataSource accepts
    # Filter out WorldModelDataset-specific params
    datasource_params = {
        'use_relative_actions', 'action_scale', 'object_name',
        'file_list', 'use_auxiliary_state',  # file_list is for CSGO DataSource
        'action_dim', 'video_suffix', 'action_suffix',  # BlockWorld DataSource
        'root', 'image_key', 'preload_trajectories', 'pad_action_dim',  # LeRobotDataSource
    }
    for key, value in kwargs.items():
        if key in datasource_params and value is not None:
            data_source_kwargs[key] = value

    data_source = create_data_source(**data_source_kwargs)

    # Create dataset
    dataset = WorldModelDataset(
        data_source=data_source,
        num_frames=num_frames,
        frame_interval=frame_interval,
        image_size=image_size,
        split=split,
        split_ratio=split_ratio,
        normalize_action=normalize_action,
        normalize_state=normalize_state,
        normalize_pixel=normalize_pixel,
        random_seed=random_seed,
        slice_mode=slice_mode,
        stride=stride,
        fixed_start_indices=fixed_start_indices,
        resize_mode=resize_mode,
    )

    return dataset


# Convenience function for creating train and val datasets together
def create_train_val_datasets(
    dataset_name: str,
    data_path: Optional[str] = None,
    data_path_train: Optional[str] = None,
    data_path_val: Optional[str] = None,
    train_file_list: Optional[str] = None,
    val_file_list: Optional[str] = None,
    val_start_indices: Optional[str] = None,
    repo_id: Optional[str] = None,
    num_frames: int = 16,
    frame_interval: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    split_ratio: float = 0.9,
    normalize_action: bool = True,
    normalize_state: bool = False,
    normalize_pixel: bool = True,
    n_rollout: Optional[int] = None,
    random_seed: int = 42,
    train_slice_mode: str = "exhaustive",  # Default to exhaustive for backward compatibility
    val_slice_mode: str = "exhaustive",
    stride: int = 1,
    resize_mode: str = "stretch",
    **kwargs
) -> Tuple[WorldModelDataset, WorldModelDataset]:
    """
    Create both train and validation datasets with shared DataSource.

    This is more efficient than creating them separately because:
    1. DataSource is created only once (shared normalization stats)
    2. Consistent train/val split using same random seed

    Args:
        train_file_list: Path to text file containing training filenames (for explicit split)
        val_file_list: Path to text file containing validation filenames (for explicit split)
        val_start_indices: Path to .npy file containing validation start frame indices (for Vid2World consistency)
        train_slice_mode: Slice mode for training ("random" recommended for efficiency)
        val_slice_mode: Slice mode for validation ("exhaustive" recommended for consistency)
        If both file lists are provided, they will be used instead of split_ratio

    Returns:
        (train_dataset, val_dataset)
    """
    import os

    def _resolve_path(path: Optional[str]) -> Optional[str]:
        """Resolve relative paths from project root when Hydra has changed cwd to run dir."""
        if path is None:
            return None
        if os.path.isabs(path) or os.path.isfile(path):
            return path
        try:
            from hydra.utils import get_original_cwd
            resolved = os.path.join(get_original_cwd(), path)
            if os.path.isfile(resolved):
                return resolved
        except Exception:
            pass
        return path

    train_file_list = _resolve_path(train_file_list)
    val_file_list = _resolve_path(val_file_list)
    val_start_indices = _resolve_path(val_start_indices)

    train_path = data_path_train or data_path
    val_path = data_path_val or data_path
    use_separate_paths = (data_path_train is not None) or (data_path_val is not None)
    use_file_lists = (train_file_list is not None) and (val_file_list is not None)

    # Load validation start indices if provided
    val_start_indices_array = None
    if val_start_indices is not None:
        val_start_indices_array = np.load(val_start_indices)
        print(f"Loaded {len(val_start_indices_array)} validation start indices from {val_start_indices}")

    # When using file lists, we create separate data sources for train and val
    if use_file_lists:
        # Create train dataset with train file list
        train_kwargs = kwargs.copy()
        train_kwargs['file_list'] = train_file_list  # Pass train file list to DataSource

        train_dataset = create_world_model_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            repo_id=repo_id,
            num_frames=num_frames,
            frame_interval=frame_interval,
            image_size=image_size,
            split="train",
            split_ratio=1.0,  # No split needed, files already filtered
            normalize_action=normalize_action,
            normalize_state=normalize_state,
            normalize_pixel=normalize_pixel,
            n_rollout=n_rollout,
            random_seed=random_seed,
            slice_mode=train_slice_mode,
            stride=stride,
            resize_mode=resize_mode,
            **train_kwargs
        )

        # Create val dataset with val file list
        val_kwargs = kwargs.copy()
        val_kwargs['file_list'] = val_file_list  # Pass val file list to DataSource

        val_dataset = create_world_model_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            repo_id=repo_id,
            num_frames=num_frames,
            frame_interval=frame_interval,
            image_size=image_size,
            split="val",
            split_ratio=0.0,  # No split needed, files already filtered
            normalize_action=normalize_action,
            normalize_state=normalize_state,
            normalize_pixel=normalize_pixel,
            n_rollout=n_rollout,
            random_seed=random_seed,
            slice_mode=val_slice_mode,
            stride=stride,
            fixed_start_indices=val_start_indices_array,
            resize_mode=resize_mode,
            **val_kwargs
        )
    # When using separate paths, create separate data sources
    elif use_separate_paths:
        train_dataset = create_world_model_dataset(
            dataset_name=dataset_name,
            data_path=train_path,
            repo_id=repo_id,
            num_frames=num_frames,
            frame_interval=frame_interval,
            image_size=image_size,
            split="train",
            split_ratio=1.0,
            normalize_action=normalize_action,
            normalize_state=normalize_state,
            normalize_pixel=normalize_pixel,
            n_rollout=n_rollout,
            random_seed=random_seed,
            slice_mode=train_slice_mode,
            stride=stride,
            resize_mode=resize_mode,
            **kwargs
        )

        val_dataset = create_world_model_dataset(
            dataset_name=dataset_name,
            data_path=val_path,
            repo_id=repo_id,
            num_frames=num_frames,
            frame_interval=frame_interval,
            image_size=image_size,
            split="val",
            split_ratio=0.0,
            normalize_action=normalize_action,
            normalize_state=normalize_state,
            normalize_pixel=normalize_pixel,
            n_rollout=n_rollout,
            random_seed=random_seed,
            slice_mode=val_slice_mode,
            stride=stride,
            fixed_start_indices=val_start_indices_array,
            resize_mode=resize_mode,
            **kwargs
        )
    # Default: use split_ratio for random split
    else:
        # Create train dataset
        train_dataset = create_world_model_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            repo_id=repo_id,
            num_frames=num_frames,
            frame_interval=frame_interval,
            image_size=image_size,
            split="train",
            split_ratio=split_ratio,
            normalize_action=normalize_action,
            normalize_state=normalize_state,
            normalize_pixel=normalize_pixel,
            n_rollout=n_rollout,
            random_seed=random_seed,
            slice_mode=train_slice_mode,
            stride=stride,
            resize_mode=resize_mode,
            **kwargs
        )

        # Share the data source and inherit train's normalization stats.
        val_dataset = WorldModelDataset(
            data_source=train_dataset.data_source,  # Reuse data source
            num_frames=num_frames,
            frame_interval=frame_interval,
            image_size=image_size,
            split="val",
            split_ratio=split_ratio,
            normalize_action=normalize_action,
            normalize_state=normalize_state,
            normalize_pixel=normalize_pixel,
            random_seed=random_seed,
            slice_mode=val_slice_mode,
            stride=stride,
            fixed_start_indices=val_start_indices_array,
            resize_mode=resize_mode,
            _inherit_stats_from=train_dataset,
        )

    return train_dataset, val_dataset


def create_eval_only_dataset(
    dataset_name: str,
    num_frames: int,
    frame_interval: int,
    image_size: Tuple[int, int],
    precomputed_slices: List[Dict[str, int]],
    data_path: Optional[str] = None,
    data_path_train: Optional[str] = None,
    data_path_val: Optional[str] = None,
    repo_id: Optional[str] = None,
    split_ratio: float = 0.9,
    normalize_action: bool = True,
    normalize_state: bool = False,
    normalize_pixel: bool = True,
    n_rollout: Optional[int] = None,
    random_seed: int = 42,
    val_slice_mode: str = "exhaustive",
    stride: int = 1,
    resize_mode: str = "stretch",
    **kwargs,
) -> WorldModelDataset:
    """Create a val-only dataset from precomputed slice specs, skipping full indexing."""
    import os

    def _resolve_path(path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if os.path.isabs(path) or os.path.isfile(path):
            return path
        try:
            from hydra.utils import get_original_cwd
            resolved = os.path.join(get_original_cwd(), path)
            if os.path.isfile(resolved):
                return resolved
        except Exception:
            pass
        return path

    val_path = data_path_val or data_path

    data_source_kwargs = {
        "dataset_name": dataset_name,
        "n_rollout": n_rollout,
    }
    if val_path is not None:
        data_source_kwargs["data_path"] = val_path
    if repo_id is not None:
        data_source_kwargs["repo_id"] = repo_id

    datasource_params = {
        'use_relative_actions', 'action_scale', 'object_name',
        'file_list', 'use_auxiliary_state',
        'action_dim', 'video_suffix', 'action_suffix',
        'root', 'image_key', 'preload_trajectories', 'pad_action_dim',
    }
    for key, value in kwargs.items():
        if key in datasource_params and value is not None:
            data_source_kwargs[key] = value

    data_source = create_data_source(**data_source_kwargs)

    return WorldModelDataset(
        data_source=data_source,
        num_frames=num_frames,
        frame_interval=frame_interval,
        image_size=image_size,
        split="val",
        split_ratio=split_ratio,
        normalize_action=normalize_action,
        normalize_state=normalize_state,
        normalize_pixel=normalize_pixel,
        random_seed=random_seed,
        slice_mode=val_slice_mode,
        stride=stride,
        resize_mode=resize_mode,
        use_data_source_stats=True,
        precomputed_slices=precomputed_slices,
    )
