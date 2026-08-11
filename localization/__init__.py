"""UAV Visual Geo-Localization via Structure-Based Triangulation.

Main pipeline entry points with lazy imports to avoid pulling in heavy deps
(torch, cv2, sklearn) until actually needed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "load_model",
    "process_uav",
    "SatelliteDatabase",
    "build_satellite_descriptors",
    "query_uav",
]


def __getattr__(name: str) -> Any:
    if name == "load_model":
        from .segmentation import load_model
        return load_model

    if name == "process_uav":
        from .preprocess.uav import process_uav
        return process_uav

    if name == "SatelliteDatabase":
        from .database.kdtree import SatelliteDatabase
        return SatelliteDatabase

    if name == "build_satellite_descriptors":
        from .database.builder import build_satellite_descriptors
        return build_satellite_descriptors

    if name == "query_uav":
        from .matching.query import query_uav
        return query_uav

    raise AttributeError(f"module 'localization' has no attribute {name!r}")
