"""Runtime config dataclasses (paper §4.6 hyperparameters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PathConfig:
    """File-system paths for one UAV-VisLoc flight."""

    data_root: Path = Path("UAV-VisLoc")
    flight_id: str = "01"
    output_dir: Path = Path("outputs")

    @property
    def flight_csv(self) -> Path:
        return self.data_root / self.flight_id / f"{self.flight_id}.csv"

    @property
    def img_dir(self) -> Path:
        return self.data_root / self.flight_id / "drone"

    @property
    def satellite_tif(self) -> Path:
        return self.data_root / self.flight_id / f"satellite{self.flight_id}.tif"

    @property
    def bounds_csv(self) -> Path:
        return self.data_root / "satellite_coordinates_range.csv"


@dataclass
class SegmentationConfig:
    score_threshold: float = 0.5
    min_polygon_area: float = 50.0
    epsilon_factor: float = 0.02
    contour_method: str = "marching_squares"


@dataclass
class IndexConfig:
    """Paper §4.6: patch_size=500, stride=100, KDTree leaf=40, K=5."""

    patch_size: int = 500
    stride: int = 100
    kdtree_leaf_size: int = 40
    top_k: int = 5


@dataclass
class TrainConfig:
    """Mask R-CNN training hyperparameters (used by scripts/train_maskrcnn.py)."""

    num_classes: int = 2
    backbone: str = "resnet50"
    pretrained: bool = True
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_epochs: int = 15
    scheduler_milestones: Optional[List[int]] = None
    scheduler_gamma: float = 0.1
    warmup_iters: int = 500
    warmup_factor: float = 1.0 / 1000
    seed: int = 42

    def __post_init__(self):
        if self.scheduler_milestones is None:
            self.scheduler_milestones = [6, 8]


@dataclass
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
