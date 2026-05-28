"""Step 1 - UAV image preprocessing.

Per paper §4.1: scale normalization by altitude vs. satellite GSD, yaw alignment
from compass heading, roll/pitch correction, central crop and resize to
500x500 px.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from PIL import Image


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
class UAVPose:
    height: float
    pitch: float
    roll: float
    yaw: float


class UAVPreprocessor:
    """UAV preprocessing pipeline (paper §4.1, Step 1).

    Steps applied in order: yaw alignment, roll correction, pitch-induced
    vertical scaling, altitude-based isotropic rescale, central crop, resize.
    """

    def __init__(self, crop_size: int = 2000, out_size: int = 500, ref_height: float = 400.0):
        self.crop_size = crop_size
        self.out_size = out_size
        self.ref_height = ref_height

    def _load_pose_row(self, img_path: str, csv_path: str) -> pd.Series:
        df = pd.read_csv(csv_path, names=VISLOC_CSV_COLUMNS, header=None, sep=r"\s+|,", engine="python")
        df["filename"] = df["filename"].astype(str).str.strip()
        filename = os.path.basename(img_path)
        row_data = df[df["filename"] == filename]
        if len(row_data) == 0:
            raise ValueError(f"Filename '{filename}' not found in CSV '{csv_path}'")
        return row_data.iloc[0]

    @staticmethod
    def _extract_pose(row: pd.Series) -> UAVPose:
        return UAVPose(
            height=float(row["height"]),
            pitch=float(row["Omega"]),
            roll=float(row["Kappa"]),
            yaw=float(row["Phi1"]),
        )

    @staticmethod
    def _build_meta(pose: UAVPose) -> Dict[str, float]:
        return {
            "height": pose.height,
            "Pitch (Omega)": pose.pitch,
            "Roll (Kappa)": pose.roll,
            "Yaw (Phi)": pose.yaw,
            "yaw_applied": pose.yaw,
        }

    def process(self, img_path: str, csv_path: str) -> Tuple[Image.Image, Image.Image, Dict[str, float]]:
        img_raw = Image.open(img_path).convert("RGB")
        row = self._load_pose_row(img_path, csv_path)
        pose = self._extract_pose(row)

        img = img_raw.copy()
        # Step 1b - yaw alignment (paper §4.1): rotate by -yaw so North is up.
        img = img.rotate(-pose.yaw, resample=Image.BICUBIC, expand=True)

        # Step 1c - roll/pitch correction.
        if abs(pose.roll) > 0.5:
            img = img.rotate(-pose.roll, resample=Image.BICUBIC, expand=True)

        pitch_factor = math.cos(math.radians(abs(pose.pitch)))
        if pitch_factor > 0.1:
            w_curr, h_curr = img.size
            new_h = int(h_curr * pitch_factor)
            img = img.crop((0, (h_curr - new_h) // 2, w_curr, (h_curr + new_h) // 2))

        # Step 1a - scale normalization (height -> reference height ~400m).
        scale = self.ref_height / max(pose.height, 1.0)
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)

        # Step 1d - central crop + resize.
        w, h = img.size
        cx, cy = w // 2, h // 2
        half = self.crop_size // 2
        left = max(cx - half, 0)
        upper = max(cy - half, 0)
        right = min(cx + half, w)
        lower = min(cy + half, h)
        img = img.crop((left, upper, right, lower))
        img_rotated = img.resize((self.out_size, self.out_size), resample=Image.BICUBIC)

        return img_raw, img_rotated, self._build_meta(pose)


def process_uav(
    img_path: str,
    csv_path: str,
    crop_size: int = 2000,
    out_size: int = 500,
    ref_height: float = 400.0,
) -> Tuple[Image.Image, Image.Image, Dict[str, float]]:
    """Functional API for the UAV preprocessing pipeline."""
    preprocessor = UAVPreprocessor(crop_size=crop_size, out_size=out_size, ref_height=ref_height)
    return preprocessor.process(img_path=img_path, csv_path=csv_path)
