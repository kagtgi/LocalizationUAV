"""Step 5b - K-NN query, plurality vote, top-N retrieval, and visualization.

``visualize`` requires OpenCV + matplotlib and is loaded lazily.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "query_uav",
    "query_top_n_patches",
    "plurality_vote",
    "PatchPrediction",
    "QueryResult",
    "draw_predicted_position",
    "draw_gt_and_prediction",
    "draw_top_n_predictions",
    "draw_gt_vs_topn_centroids",
    "crop_zoom_around",
    "render_match_figure",
    "render_localization_result",
    "render_top_n_result",
]


def __getattr__(name: str) -> Any:
    if name in ("query_uav", "query_top_n_patches", "plurality_vote", "PatchPrediction", "QueryResult"):
        from . import query as _q

        return getattr(_q, name)
    if name in (
        "draw_predicted_position",
        "draw_gt_and_prediction",
        "draw_top_n_predictions",
        "draw_gt_vs_topn_centroids",
        "crop_zoom_around",
        "render_match_figure",
        "render_localization_result",
        "render_top_n_result",
    ):
        from . import visualize as _v

        return getattr(_v, name)
    raise AttributeError(f"module 'localization.matching' has no attribute {name!r}")
