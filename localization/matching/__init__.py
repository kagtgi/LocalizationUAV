"""Step 5b - K-NN query, plurality vote, and result visualization.

``visualize`` requires OpenCV + matplotlib and is loaded lazily.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "query_uav",
    "plurality_vote",
    "draw_predicted_position",
    "draw_gt_and_prediction",
    "crop_zoom_around",
    "render_match_figure",
    "render_localization_result",
]


def __getattr__(name: str) -> Any:
    if name in ("query_uav", "plurality_vote"):
        from . import query as _q

        return getattr(_q, name)
    if name in (
        "draw_predicted_position",
        "draw_gt_and_prediction",
        "crop_zoom_around",
        "render_match_figure",
        "render_localization_result",
    ):
        from . import visualize as _v

        return getattr(_v, name)
    raise AttributeError(f"module 'localization.matching' has no attribute {name!r}")
