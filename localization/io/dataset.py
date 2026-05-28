"""UAV-VisLoc dataset helpers.

Expected on-disk layout (configurable; see ``UAV-VisLoc/`` in the repo root):

    UAV-VisLoc/
        satellite_coordinates_range.csv
        01/
            drone/01_0001.JPG ...
            satellite01.tif
            01.csv
        02/ ...
        ...
        11/ ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


VISLOC_CSV_COLUMNS = [
    "num",
    "filename",
    "date",
    "lat",
    "lon",
    "height",
    "Omega",
    "Kappa",
    "Phi1",
    "Phi2",
]


@dataclass
class VisLocFlight:
    """One UAV-VisLoc flight (e.g. ``01``)."""

    flight_id: str
    root: Path

    @property
    def flight_dir(self) -> Path:
        return self.root / self.flight_id

    @property
    def drone_dir(self) -> Path:
        return self.flight_dir / "drone"

    @property
    def satellite_tif(self) -> Path:
        return self.flight_dir / f"satellite{self.flight_id}.tif"

    @property
    def metadata_csv(self) -> Path:
        return self.flight_dir / f"{self.flight_id}.csv"

    @property
    def bounds_csv(self) -> Path:
        return self.root / "satellite_coordinates_range.csv"

    def drone_image_path(self, filename: str) -> Path:
        return self.drone_dir / filename

    def list_drone_images(self) -> list[Path]:
        if not self.drone_dir.exists():
            return []
        return sorted(p for p in self.drone_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def load_metadata(self) -> pd.DataFrame:
        return load_flight_metadata(self.metadata_csv)


def load_flight_metadata(csv_path: os.PathLike) -> pd.DataFrame:
    """Load per-image GPS+heading metadata for a single flight."""
    df = pd.read_csv(csv_path, names=VISLOC_CSV_COLUMNS, header=None, sep=r"\s+|,", engine="python")
    df["filename"] = df["filename"].astype(str).str.strip()
    return df


def get_image_pose(df: pd.DataFrame, image_filename: str) -> Optional[pd.Series]:
    """Look up the row for a given drone image filename."""
    matches = df[df["filename"] == os.path.basename(image_filename)]
    if len(matches) == 0:
        return None
    return matches.iloc[0]
