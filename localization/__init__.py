"""Structure-Based UAV Visual Geo-Localization (paper_draft/main.tex).

Top-level re-exports for the most commonly used public API. Torch-dependent
symbols (``load_model``, ``process_uav`` etc.) are loaded lazily so that
torch-free callers (e.g. geometry-only / KDTree-only paths) can ``import
localization.geometry`` or ``import localization.database`` without paying
the torch import cost.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "load_model",
    "process_uav",
    "UAVPreprocessor",
    "SatelliteDatabase",
    "build_satellite_descriptors",
    "query_uav",
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for top-level re-exports."""
    if name in ("load_model",):
        from .segmentation.model import load_model

        return load_model
    if name in ("process_uav", "UAVPreprocessor"):
        from .preprocess.uav import process_uav, UAVPreprocessor

        return {"process_uav": process_uav, "UAVPreprocessor": UAVPreprocessor}[name]
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
