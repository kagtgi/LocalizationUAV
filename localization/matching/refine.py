"""Step 6 - RANSAC sub-patch position refinement (paper Sec. 5.7).

The plurality vote (``query.py``) resolves a UAV query to one winning
satellite patch; its output position is that patch's own pre-computed
centroid, discretized at the patch stride. This module refines that estimate
using the triangle-centroid correspondences that produced the winning vote:
Step 1 already normalizes heading and scale, so the residual UAV-to-satellite
misalignment is well modeled as a pure 2-D translation, which a small RANSAC
pass over the correspondences recovers robustly against individual false
matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..database.kdtree import SatelliteDatabase


@dataclass
class RefineResult:
    pixel_xy: Tuple[float, float]
    translation: Tuple[float, float]
    n_correspondences: int
    n_inliers: int
    refined: bool

    def to_dict(self) -> dict:
        return {
            "pixel_xy": (float(self.pixel_xy[0]), float(self.pixel_xy[1])),
            "translation": (float(self.translation[0]), float(self.translation[1])),
            "n_correspondences": int(self.n_correspondences),
            "n_inliers": int(self.n_inliers),
            "refined": bool(self.refined),
        }


def ransac_refine_position(
    uav_centroids: np.ndarray,
    nearest_indices: np.ndarray,
    db: SatelliteDatabase,
    winner_code: int,
    uav_reference_xy: Tuple[float, float],
    coarse_pixel_xy: Tuple[float, float],
    inlier_threshold_px: float = 15.0,
    min_correspondences: int = 3,
) -> RefineResult:
    """Refine the winning patch's coarse centroid via RANSAC translation alignment.

    ``uav_centroids`` (M, 2) are the UAV triangle centroids in the
    preprocessed 500x500 frame, row-aligned with the ``uav_descriptors``
    passed to :func:`~localization.matching.query.query_uav` (so row ``i``
    here matches row ``i`` of ``nearest_indices``, shape (M, k)).

    For every (i, kk) whose matched satellite row falls in the winning patch,
    the pair ``(uav_centroids[i], db.centroids[nearest_indices[i, kk]])``
    implies a translation ``t = sat_centroid - uav_centroid`` (Step 1 already
    removes heading/scale, so translation is the correct residual-alignment
    model). A 1-point RANSAC - each correspondence's own ``t`` is a complete
    hypothesis - picks the translation with the most agreeing correspondences
    within ``inlier_threshold_px``, then averages that inlier set.

    ``uav_reference_xy`` is the UAV image's own nadir point (image center in
    the 500x500 frame); the refined position is that point carried through
    the estimated translation into the satellite pixel frame.

    Falls back to ``coarse_pixel_xy`` unchanged (``refined=False``) when
    fewer than ``min_correspondences`` pairs are available - refinement never
    makes the estimate worse than the coarse patch centroid.
    """
    winner_mask = db.patch_id_codes[nearest_indices] == int(winner_code)  # (M, k) bool
    rows, cols = np.nonzero(winner_mask)
    if rows.size < min_correspondences:
        return RefineResult(
            pixel_xy=(float(coarse_pixel_xy[0]), float(coarse_pixel_xy[1])),
            translation=(0.0, 0.0),
            n_correspondences=int(rows.size),
            n_inliers=0,
            refined=False,
        )

    sat_idx = nearest_indices[rows, cols]
    sat_xy = db.centroids[sat_idx].astype(np.float64)     # (n, 2)
    uav_xy = np.asarray(uav_centroids, dtype=np.float64)[rows]  # (n, 2)
    translations = sat_xy - uav_xy                         # (n, 2): one hypothesis per pair

    # 1-point RANSAC: score each hypothesis by how many other correspondences
    # agree with it within inlier_threshold_px, keep the best-supported one.
    diffs = translations[:, None, :] - translations[None, :, :]  # (n, n, 2)
    dist = np.linalg.norm(diffs, axis=2)                          # (n, n)
    inlier_counts = (dist <= float(inlier_threshold_px)).sum(axis=1)
    best = int(np.argmax(inlier_counts))
    inlier_mask = dist[best] <= float(inlier_threshold_px)
    t_hat = translations[inlier_mask].mean(axis=0)

    refined_xy = (
        float(uav_reference_xy[0]) + float(t_hat[0]),
        float(uav_reference_xy[1]) + float(t_hat[1]),
    )
    return RefineResult(
        pixel_xy=refined_xy,
        translation=(float(t_hat[0]), float(t_hat[1])),
        n_correspondences=int(rows.size),
        n_inliers=int(inlier_mask.sum()),
        refined=True,
    )
