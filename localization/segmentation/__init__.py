"""Building segmentation (Step 2): Mask R-CNN + contour + polygon.

Torch / OpenCV / scikit-image are imported lazily on first attribute access so
that the geometry sub-package can be imported without these heavy deps.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "load_model",
    "get_model",
    "extract_contours_from_mask",
    "contour_to_polygon",
    "contour_to_polygon_dynamic",
    "filter_large_polygons",
    "filter_large_polygons_dynamic",
    "segment_image",
    "segment_batch",
    "segment_image_patchwise",
    "polygons_from_soft_mask",
    "visualize_segmented_buildings",
]


def __getattr__(name: str) -> Any:
    if name in ("load_model", "get_model"):
        from .model import get_model, load_model

        return {"load_model": load_model, "get_model": get_model}[name]
    if name in (
        "extract_contours_from_mask",
        "contour_to_polygon",
        "contour_to_polygon_dynamic",
        "filter_large_polygons",
        "filter_large_polygons_dynamic",
    ):
        from . import contours as _c

        return getattr(_c, name)
    if name in (
        "segment_image",
        "segment_batch",
        "segment_image_patchwise",
        "polygons_from_soft_mask",
        "visualize_segmented_buildings",
    ):
        from . import inference as _i

        return getattr(_i, name)
    raise AttributeError(f"module 'localization.segmentation' has no attribute {name!r}")
