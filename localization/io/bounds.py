"""Satellite bounds <-> lat/lon <-> pixel conversion (paper Eq. 4)."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


EARTH_RADIUS_M = 6378137.0


def _default_bounds_csv_candidates() -> list[Path]:
    """Likely locations for ``satellite_coordinates_range.csv``."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "UAV-VisLoc" / "satellite_coordinates_range.csv",
        repo_root / "UAV-VisLoc" / "satellite_ coordinates_range.csv",
    ]
    return candidates


def _normalize_satellite_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(name)))[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _extract_numeric_id(name: str) -> Optional[str]:
    norm = _normalize_satellite_name(name)
    matches = re.findall(r"\d+", norm)
    if not matches:
        return None
    return matches[-1].lstrip("0") or "0"


def _resolve_bounds_csv_path(csv_path: Optional[str]) -> Optional[Path]:
    candidates = [Path(csv_path)] if csv_path else _default_bounds_csv_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _match_bounds_row(df: pd.DataFrame, satellite_filename: str):
    map_col = df.columns[0]
    map_names = df[map_col].astype(str).fillna("")
    map_norm = map_names.map(_normalize_satellite_name)
    map_norm_no_prefix = map_norm.str.replace(r"^satellite", "", regex=True)

    target_norm = _normalize_satellite_name(satellite_filename)
    target_no_prefix = re.sub(r"^satellite", "", target_norm)
    target_numeric = _extract_numeric_id(satellite_filename)

    exact = (
        (map_norm == target_norm)
        | (map_norm == target_no_prefix)
        | (map_norm_no_prefix == target_norm)
        | (map_norm_no_prefix == target_no_prefix)
    )
    if exact.any():
        return df.loc[exact].iloc[0], map_col

    if target_numeric is not None:
        numeric_match = map_names.map(_extract_numeric_id) == target_numeric
        if numeric_match.any():
            return df.loc[numeric_match].iloc[0], map_col

    contains = (
        map_norm.str.contains(target_norm, na=False)
        | map_norm.str.contains(target_no_prefix, na=False)
        | map_norm_no_prefix.str.contains(target_no_prefix, na=False)
    )
    if contains.any():
        return df.loc[contains].iloc[0], map_col

    return None, map_col


def load_satellite_bounds(satellite_filename: str, csv_path: Optional[str] = None) -> Optional[dict]:
    """Load ``LT_lat / LT_lon / RB_lat / RB_lon`` for a satellite image."""
    resolved = _resolve_bounds_csv_path(csv_path)
    if resolved is None:
        return None

    df = pd.read_csv(resolved)
    row, _ = _match_bounds_row(df, satellite_filename)
    if row is None:
        return None
    return {
        "LT_lat": float(row["LT_lat_map"]),
        "LT_lon": float(row["LT_lon_map"]),
        "RB_lat": float(row["RB_lat_map"]),
        "RB_lon": float(row["RB_lon_map"]),
    }


def latlon_to_pixel(
    lat: float,
    lon: float,
    bounds: dict,
    sat_width: int,
    sat_height: int,
    clip: bool = True,
) -> Tuple[int, int]:
    """Convert ``(lat, lon)`` to a pixel ``(x, y)`` on a north-up satellite image."""
    lon_span = bounds["RB_lon"] - bounds["LT_lon"]
    lat_span = bounds["LT_lat"] - bounds["RB_lat"]
    if lon_span == 0 or lat_span == 0:
        raise ValueError("Invalid satellite bounds: zero span")

    x = int(round((lon - bounds["LT_lon"]) / lon_span * max(sat_width - 1, 1)))
    y = int(round((bounds["LT_lat"] - lat) / lat_span * max(sat_height - 1, 1)))

    if clip:
        x = max(0, min(x, sat_width - 1))
        y = max(0, min(y, sat_height - 1))
    return x, y


def pixel_to_latlon(
    px: float,
    py: float,
    bounds: dict,
    sat_width: int,
    sat_height: int,
) -> Tuple[float, float]:
    """Inverse of :func:`latlon_to_pixel` for north-up rectangular bounds."""
    lon_span = bounds["RB_lon"] - bounds["LT_lon"]
    lat_span = bounds["LT_lat"] - bounds["RB_lat"]
    fx = float(px) / max(sat_width - 1, 1)
    fy = float(py) / max(sat_height - 1, 1)
    lat = bounds["LT_lat"] - fy * lat_span
    lon = bounds["LT_lon"] + fx * lon_span
    return float(lat), float(lon)


def estimate_satellite_resolution_meters(bounds: dict, sat_width: int, sat_height: int) -> dict:
    """Approximate ground resolution from rectangular lat/lon bounds."""
    lt_lat, lt_lon = float(bounds["LT_lat"]), float(bounds["LT_lon"])
    rb_lat, rb_lon = float(bounds["RB_lat"]), float(bounds["RB_lon"])

    meters_per_deg_lat = math.pi * EARTH_RADIUS_M / 180.0
    center_lat = 0.5 * (lt_lat + rb_lat)
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))

    width_m = abs(rb_lon - lt_lon) * meters_per_deg_lon
    height_m = abs(lt_lat - rb_lat) * meters_per_deg_lat
    denom_x = max(int(sat_width) - 1, 1)
    denom_y = max(int(sat_height) - 1, 1)
    return {
        "width_m": width_m,
        "height_m": height_m,
        "center_lat": center_lat,
        "m_per_px_x": width_m / denom_x,
        "m_per_px_y": height_m / denom_y,
        "m_per_px_mean": 0.5 * ((width_m / denom_x) + (height_m / denom_y)),
    }


def pixel_offset_to_meters(dx_px: float, dy_px: float, bounds: dict, sat_width: int, sat_height: int) -> dict:
    """Convert a pixel offset to ground meters via :func:`estimate_satellite_resolution_meters`."""
    res = estimate_satellite_resolution_meters(bounds, sat_width, sat_height)
    dx_m = float(dx_px) * res["m_per_px_x"]
    dy_m = float(dy_px) * res["m_per_px_y"]
    return {
        **res,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "distance_m": math.hypot(dx_m, dy_m),
    }
