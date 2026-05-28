import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class UAVPose:
    height: float
    pitch: float
    roll: float
    yaw: float


class UAVPreprocessor:
    """Class-based wrapper for UAV preprocessing while preserving legacy logic."""

    def __init__(self, crop_size: int = 2000, out_size: int = 500, ref_height: float = 400.0):
        self.crop_size = crop_size
        self.out_size = out_size
        self.ref_height = ref_height

    def _load_pose_row(self, img_path: str, csv_path: str) -> pd.Series:
        col_names = ["num", "filename", "date", "lat", "lon", "height", "Omega", "Kappa", "Phi1", "Phi2"]
        df = pd.read_csv(csv_path, names=col_names, header=None, sep=r"\s+|,", engine="python")
        filename = os.path.basename(img_path)
        row_data = df[df["filename"] == filename]
        if len(row_data) == 0:
            raise ValueError(f"Khong tim thay file '{filename}' trong CSV '{csv_path}'")
        return row_data.iloc[0]

    def _extract_pose(self, row: pd.Series) -> UAVPose:
        return UAVPose(
            height=float(row["height"]),
            pitch=float(row["Omega"]),
            roll=float(row["Kappa"]),
            yaw=float(row["Phi1"]),
        )

    def _build_meta(self, pose: UAVPose) -> Dict[str, float]:
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
        print(f"   [process_uav] Xoay anh UAV {os.path.basename(img_path)}: yaw={pose.yaw:.1f} deg")
        img = img.rotate(-pose.yaw, resample=Image.BICUBIC, expand=True)

        if abs(pose.roll) > 0.5:
            img = img.rotate(-pose.roll, resample=Image.BICUBIC, expand=True)

        pitch_factor = math.cos(math.radians(abs(pose.pitch)))
        if pitch_factor > 0.1:
            w_curr, h_curr = img.size
            new_h = int(h_curr * pitch_factor)
            img = img.crop((0, (h_curr - new_h) // 2, w_curr, (h_curr + new_h) // 2))

        scale = self.ref_height / max(pose.height, 1.0)
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)

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
):
    """Backward-compatible functional API."""
    preprocessor = UAVPreprocessor(crop_size=crop_size, out_size=out_size, ref_height=ref_height)
    return preprocessor.process(img_path=img_path, csv_path=csv_path)


class VisualLocDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        col_names = ["num", "filename", "date", "lat", "lon", "height", "Omega", "Kappa", "Phi1", "Phi2"]
        try:
            self.annotations = pd.read_csv(
                csv_file,
                names=col_names,
                header=None,
                sep=r"\s+|,",
                engine="python",
            )
            self.annotations["filename"] = self.annotations["filename"].astype(str).str.strip()
            print(f"-> Da tai {len(self.annotations)} dong du lieu tu CSV.")
        except Exception as e:
            print(f"Loi doc CSV: {e}")
            self.annotations = pd.DataFrame()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        img_name = row["filename"]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Canh bao: Khong tim thay anh {img_path}")
            dummy_img = torch.zeros((3, 500, 500))
            dummy_target = torch.zeros(3)
            return dummy_img, dummy_target

        try:
            # Keep legacy call style to preserve existing behavior.
            image = process_uav(img_path, row)
        except Exception:
            image = Image.open(img_path).convert("RGB").resize((500, 500))

        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
            height = float(row["height"])
            target = torch.tensor([lat, lon, height], dtype=torch.float32)
        except Exception:
            target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


class UAVDataModule:
    """OOP wrapper to create dataloaders with the same defaults as legacy code."""

    def __init__(self, config):
        self.config = config

    def get_dataloader(self, batch_size: int = 4, shuffle: bool = True):
        data_transform = transforms.Compose(
            [
                transforms.Resize((500, 500)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = VisualLocDataset(
            csv_file=self.config.get_csv_path(),
            root_dir=self.config.get_img_dir(),
            transform=data_transform,
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def get_uav_dataloader(config, batch_size=4, shuffle=True):
    """Backward-compatible functional API."""
    return UAVDataModule(config).get_dataloader(batch_size=batch_size, shuffle=shuffle)
