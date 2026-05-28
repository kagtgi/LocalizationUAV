"""Step 5a - satellite descriptor database (patches + KDTree).

Lazy imports keep ``localization.database.kdtree`` torch-free; only
``build_satellite_descriptors`` pulls in the segmentation backbone.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "iter_patches",
    "patch_count",
    "build_satellite_descriptors",
    "extract_patch_descriptors",
    "SatelliteDatabase",
]


def __getattr__(name: str) -> Any:
    if name in ("iter_patches", "patch_count"):
        from . import patches as _p

        return getattr(_p, name)
    if name in ("build_satellite_descriptors", "extract_patch_descriptors"):
        from . import builder as _b

        return getattr(_b, name)
    if name == "SatelliteDatabase":
        from .kdtree import SatelliteDatabase

        return SatelliteDatabase
    raise AttributeError(f"module 'localization.database' has no attribute {name!r}")
