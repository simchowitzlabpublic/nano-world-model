"""
FlowM BlockWorld data source for 3D memory benchmark episodes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import DataSource, TrajectoryData


class BlockWorldDataSource(DataSource):
    """
    Data source for FlowM Dynamic BlockWorld episodes.

    Expected directory structure:
        data_path/
        ├── sunday_v2_training/
        │   └── 0/
        │       ├── 0000_rgb.mp4
        │       ├── 0000_depth.mp4
        │       └── 0000_actions.pt
        ├── sunday_v2_validation/
        ├── sunday_v2_static_training/
        ├── sunday_v2_static_validation/
        ├── tex_training/
        └── tex_validation/

    The companion action file is a torch object with an ``actions`` entry.
    Integer actions are converted to one-hot vectors to match the
    nano-world-model action-conditioning interface.
    """

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        action_dim: int = 5,
        file_list: Optional[str] = None,
        video_suffix: str = "_rgb.mp4",
        action_suffix: str = "_actions.pt",
    ) -> None:
        """Create a BlockWorld data source from a split or dataset root."""
        self.data_path = Path(data_path).expanduser()
        self._action_dim = int(action_dim)
        self.file_list = file_list
        self.video_suffix = video_suffix
        self.action_suffix = action_suffix

        if self._action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self._action_dim}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"BlockWorld data path not found: {self.data_path}")

        self._episodes = self._discover_episodes(file_list=file_list)
        if n_rollout is not None:
            self._episodes = self._episodes[:n_rollout]
        if not self._episodes:
            raise FileNotFoundError(
                f"No BlockWorld videos matching '*{self.video_suffix}' found in {self.data_path}"
            )

        self._action_cache: Dict[int, torch.Tensor] = {}

        print(f"Loaded {len(self._episodes)} BlockWorld episodes")
        print(f"  Action dim: {self._action_dim}, State dim: 0 (pure vision)")

    def _discover_episodes(self, file_list: Optional[str]) -> List[Tuple[str, Path, Path]]:
        """Find RGB videos and their companion action files."""
        if file_list is None:
            video_paths = sorted(self.data_path.rglob(f"*{self.video_suffix}"), key=str)
        else:
            video_paths = self._load_file_list(file_list)

        episodes: List[Tuple[str, Path, Path]] = []
        for video_path in video_paths:
            action_path = self._action_path_from_video(video_path)
            if not action_path.exists():
                raise FileNotFoundError(
                    f"Missing BlockWorld action file for {video_path}: expected {action_path}"
                )
            relative_path = video_path.relative_to(self.data_path).as_posix()
            episodes.append((relative_path, video_path, action_path))
        return episodes

    def _load_file_list(self, file_list: str) -> List[Path]:
        """Load explicit video paths from a file list."""
        file_list_path = Path(file_list).expanduser()
        if not file_list_path.is_absolute():
            file_list_path = Path.cwd() / file_list_path
        if not file_list_path.exists():
            raise FileNotFoundError(f"BlockWorld file list not found: {file_list_path}")

        video_paths: List[Path] = []
        with open(file_list_path, "r") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                path = Path(line)
                if not path.is_absolute():
                    path = self.data_path / path
                video_paths.append(path)
        return sorted(video_paths, key=str)

    def _action_path_from_video(self, video_path: Path) -> Path:
        """Return the FlowM action sidecar path for an RGB video path."""
        stem = video_path.stem
        if stem.endswith("_rgb"):
            stem = stem[:-4]
        return video_path.with_name(f"{stem}{self.action_suffix}")

    def _load_actions(self, index: int) -> torch.Tensor:
        """Load and cache one episode's action sequence."""
        if index not in self._action_cache:
            _, _, action_path = self._episodes[index]
            payload = torch.load(action_path, map_location=torch.device("cpu"), weights_only=False)
            if isinstance(payload, dict):
                if "actions" not in payload:
                    raise KeyError(f"BlockWorld action file has no 'actions' key: {action_path}")
                actions = payload["actions"]
            else:
                actions = payload

            self._action_cache[index] = self._format_actions(actions, action_path)
        return self._action_cache[index]

    def _format_actions(self, actions: torch.Tensor, action_path: Path) -> torch.Tensor:
        """Convert integer or vector actions into float action vectors."""
        actions = torch.as_tensor(actions)
        if actions.ndim == 1:
            if actions.numel() == 0:
                raise ValueError(f"BlockWorld action file is empty: {action_path}")
            action_ids = actions.long()
            min_action = int(action_ids.min().item())
            max_action = int(action_ids.max().item())
            if min_action < 0 or max_action >= self._action_dim:
                raise ValueError(
                    f"BlockWorld actions in {action_path} must be in [0, {self._action_dim}); "
                    f"found [{min_action}, {max_action}]"
                )
            return F.one_hot(action_ids, num_classes=self._action_dim).float()

        if actions.ndim == 2:
            if actions.shape[-1] == 1:
                return self._format_actions(actions.squeeze(-1), action_path)
            if actions.shape[-1] != self._action_dim:
                raise ValueError(
                    f"BlockWorld action vectors in {action_path} have dim {actions.shape[-1]}, "
                    f"expected {self._action_dim}"
                )
            return actions.float()

        raise ValueError(
            f"BlockWorld actions in {action_path} must have shape [T] or [T, {self._action_dim}], "
            f"got {tuple(actions.shape)}"
        )

    def load_trajectory(self, index: int) -> TrajectoryData:
        """Load metadata and actions for the trajectory at the given index."""
        if index >= len(self._episodes):
            raise IndexError(f"Index {index} out of range [0, {len(self._episodes)})")

        actions = self._load_actions(index)
        states = torch.zeros(actions.shape[0], 0, dtype=torch.float32)
        relative_path, _, _ = self._episodes[index]
        return TrajectoryData(
            states=states,
            actions=actions,
            seq_length=actions.shape[0],
            meta={"episode_id": relative_path},
        )

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1,
    ) -> torch.Tensor:
        """Load RGB frames for a specific trajectory and time range."""
        if index >= len(self._episodes):
            raise IndexError(f"Index {index} out of range [0, {len(self._episodes)})")

        _, video_path, _ = self._episodes[index]
        frame_indices = list(range(start, end, step))
        try:
            from decord import VideoReader, cpu

            reader = VideoReader(str(video_path), ctx=cpu(0))
            if frame_indices and frame_indices[-1] >= len(reader):
                raise ValueError(
                    f"Frame index {frame_indices[-1]} exceeds video length {len(reader)} for {video_path}"
                )

            frames = reader.get_batch(frame_indices)
            if not isinstance(frames, torch.Tensor):
                if hasattr(frames, "asnumpy"):
                    frames = frames.asnumpy()
                frames = torch.as_tensor(frames)
        except ImportError:
            frames = self._load_visual_frames_with_imageio(video_path, frame_indices)
        return frames.float().permute(0, 3, 1, 2).contiguous() / 255.0

    def _load_visual_frames_with_imageio(self, video_path: Path, frame_indices: List[int]) -> torch.Tensor:
        """Load RGB frames with imageio when decord is unavailable."""
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "BlockWorld video loading requires decord or imageio. Install the "
                "nano-world-model environment before training or evaluation."
            ) from exc

        reader = imageio.get_reader(video_path)
        try:
            frames = [torch.as_tensor(reader.get_data(i)) for i in frame_indices]
        finally:
            reader.close()
        return torch.stack(frames, dim=0)

    def get_num_trajectories(self) -> int:
        """Return total number of trajectories in this data source."""
        return len(self._episodes)

    def get_seq_length(self, index: int) -> int:
        """Return sequence length without loading video frames."""
        if index >= len(self._episodes):
            raise IndexError(f"Index {index} out of range [0, {len(self._episodes)})")
        return int(self._load_actions(index).shape[0])

    @property
    def action_dim(self) -> int:
        """Return the one-hot BlockWorld action dimension."""
        return self._action_dim

    @property
    def state_dim(self) -> int:
        """Return the state dimension; BlockWorld is exposed as pure vision."""
        return 0
