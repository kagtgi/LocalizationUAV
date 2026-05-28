import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PathConfig:
    data_root: Path
    flight_index: str = '01'
    output_dir: Path = Path('outputs')

    @property
    def flight_csv(self) -> Path:
        return self.data_root / self.flight_index / f'{self.flight_index}.csv'

    @property
    def img_dir(self) -> Path:
        return self.data_root / self.flight_index / 'drone'

    # Backward-compatible accessors
    def get_img_dir(self):
        return str(self.img_dir)

    def get_csv_path(self):
        return str(self.flight_csv)


@dataclass
class MatchingConfig:
    top_k: int = 5
    metric: str = 'complex'
    vm_kappa: float = 4.0
    linear_weight: float = 1.0
    include_shape_features: bool = False


@dataclass
class SegmentationConfig:
    score_threshold: float = 0.5
    method: str = 'marching_squares'
    max_search_radius: float = 100.0


@dataclass
class AppConfig:
    paths: PathConfig
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)


@dataclass
class Config:
    """Legacy config API retained for existing code paths."""

    data_root: str = r'D:\1-REFERENCES\12-IMACS\UAV_nonGPS\UAV_nonGPS_dataset'
    flight_index: str = '01'

    num_classes: int = 2
    backbone: str = 'resnet50'
    pretrained: bool = True

    batch_size: int = 4
    num_workers: int = 2

    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005

    num_epochs: int = 15
    print_freq: int = 10

    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5

    scheduler_milestones: Optional[List[int]] = None
    scheduler_gamma: float = 0.1

    warmup_iters: int = 500
    warmup_factor: float = 1.0 / 1000

    seed: int = 42

    def __post_init__(self):
        if self.scheduler_milestones is None:
            self.scheduler_milestones = [6, 8]

    def get_img_dir(self):
        return os.path.join(self.data_root, self.flight_index, 'drone')

    def get_csv_path(self):
        return os.path.join(self.data_root, self.flight_index, f'{self.flight_index}.csv')


config = Config()
