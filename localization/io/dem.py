"""Coarse public DEM (AWS Open Data 'elevation-tiles-prod', terrarium encoding).

Used only to convert UAV-VisLoc heights, which are above sea level, into
heights above ground: agl = h - z_ground. z_ground is the median DEM elevation
inside the prior search window (INS/VO prior), never at the ground-truth
position. Tiles are cached locally; elevation = R*256 + G + B/256 - 32768 (m).
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


class DEM:
    def __init__(self, cache_dir: str | Path, zoom: int = 12):
        self.dir = Path(cache_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.z = zoom
        self._tiles = {}

    def _tile(self, tx, ty):
        key = (tx, ty)
        if key not in self._tiles:
            f = self.dir / f"{self.z}_{tx}_{ty}.png"
            if not f.exists():
                urllib.request.urlretrieve(URL.format(z=self.z, x=tx, y=ty), f)
            a = np.asarray(Image.open(f).convert("RGB"), np.float64)
            self._tiles[key] = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
        return self._tiles[key]

    def elevation(self, lat: float, lon: float) -> float:
        x, y = _tile_xy(lat, lon, self.z)
        tx, ty = int(x), int(y)
        t = self._tile(tx, ty)
        px, py = (x - tx) * 256, (y - ty) * 256
        i, j = min(int(py), 255), min(int(px), 255)
        return float(t[i, j])

    def window_median(self, lat: float, lon: float, radius_m: float, n: int = 15) -> float:
        """Median elevation on an n x n grid inside a disk of radius_m around (lat, lon)."""
        dlat = radius_m / 111132.954
        dlon = radius_m / (111412.84 * math.cos(math.radians(lat)))
        vals = []
        for a in np.linspace(-1, 1, n):
            for b in np.linspace(-1, 1, n):
                if a * a + b * b <= 1:
                    vals.append(self.elevation(lat + a * dlat, lon + b * dlon))
        return float(np.median(vals))
