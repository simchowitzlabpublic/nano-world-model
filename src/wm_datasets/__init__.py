"""
Datasets module for world model training.

This module provides a unified data loading pipeline with three layers:
1. DataSource: Low-level data access (files, APIs)
2. WorldModelDataset: PyTorch Dataset with slicing and normalization
3. Registry: Factory functions for easy instantiation
"""

from .data_source import (
    DataSource,
    BlockWorldDataSource,
    DinoWorldModelDataSource,
    LeRobotDataSource,
    TrajectoryData,
    create_data_source,
)

from .world_model_dataset import (
    WorldModelDataset,
    SliceInfo,
    create_world_model_dataset,
    create_train_val_datasets,
)

__all__ = [
    # DataSource layer
    "DataSource",
    "BlockWorldDataSource",
    "DinoWorldModelDataSource",
    "LeRobotDataSource",
    "TrajectoryData",
    "create_data_source",

    # WorldModelDataset layer
    "WorldModelDataset",
    "SliceInfo",
    "create_world_model_dataset",
    "create_train_val_datasets",

]
