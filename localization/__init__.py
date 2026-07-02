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
    "query_top_n_patches",
    "ransac_refine_position",
    "draw_gt_vs_topn_centroids",
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
    if name in ("query_uav", "query_top_n_patches"):
        from .matching import query as _q

        return getattr(_q, name)
    if name == "ransac_refine_position":
        from .matching.refine import ransac_refine_position

        return ransac_refine_position
    if name == "draw_gt_vs_topn_centroids":
        from .matching.visualize import draw_gt_vs_topn_centroids

        return draw_gt_vs_topn_centroids
    raise AttributeError(f"module 'localization' has no attribute {name!r}")
