"""
Factory helpers for DataSource creation.
"""

from typing import Optional

from .dino_wm import DinoWorldModelDataSource, PushTDataSource, DeformableEnvDataSource
from .lerobot import LeRobotDataSource
from .game import CSGODataSource
from .memory import BlockWorldDataSource
from .base import DataSource


def create_data_source(
    dataset_name: str,
    data_path: str,
    n_rollout: Optional[int] = None,
    **kwargs
) -> DataSource:
    """
    Factory function to create the appropriate data source based on dataset name.
    """
    if dataset_name in ["point_maze", "wall"]:
        return DinoWorldModelDataSource(
            data_path=data_path,
            video_format="pth",
            n_rollout=n_rollout,
            **kwargs
        )

    if dataset_name == "pusht":
        if "use_relative_actions" not in kwargs:
            raise ValueError(
                "PushT dataset requires explicit 'use_relative_actions' parameter in config. "
                "Set use_relative_actions: true/false in your config file."
            )
        return PushTDataSource(
            data_path=data_path,
            n_rollout=n_rollout,
            **kwargs
        )

    if dataset_name in ["rope", "granular"]:
        if "object_name" not in kwargs or kwargs["object_name"] is None:
            kwargs["object_name"] = dataset_name
        return DeformableEnvDataSource(
            data_path=data_path,
            n_rollout=n_rollout,
            **kwargs
        )

    if dataset_name == "rt1":
        # RT-1 is consumed as the lerobot HF dataset (fractal20220817_data).
        # `data_path` is the HF repo_id; optional `root` kwarg points to a
        # local mirror. No per-episode .pth format is supported.
        lerobot_params = {'root', 'image_key', 'preload_trajectories', 'episodes'}
        lerobot_kwargs = {k: v for k, v in kwargs.items() if k in lerobot_params}
        return LeRobotDataSource(
            repo_id=data_path,
            n_rollout=n_rollout,
            **lerobot_kwargs,
        )

    if dataset_name == "csgo":
        # Filter kwargs - CSGO only supports file_list and use_auxiliary_state
        csgo_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['file_list', 'use_auxiliary_state']
        }
        return CSGODataSource(
            data_path=data_path,
            n_rollout=n_rollout,
            **csgo_kwargs
        )

    if dataset_name == "blockworld":
        blockworld_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['action_dim', 'file_list', 'video_suffix', 'action_suffix']
        }
        return BlockWorldDataSource(
            data_path=data_path,
            n_rollout=n_rollout,
            **blockworld_kwargs,
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"Supported: point_maze, wall, pusht, rope, granular, rt1, csgo, blockworld"
    )
