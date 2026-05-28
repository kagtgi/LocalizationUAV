"""Brute-force O(M*N) matcher - NOT used by the production pipeline.

Preserved from the original ``matching/features.py`` for reference. The paper
specifies a ``scipy.spatial.KDTree`` with ell_1 metric and plurality voting;
see ``localization/database/kdtree.py`` and ``localization/matching/query.py``.
"""

from __future__ import annotations

import heapq
import math
from typing import List

import pandas as pd

from .extra_distance_metrics import (
    circular_mean_distance,
    complex_angle_distance,
    geodesic_feature_distance,
    l1_distance,
    l2_distance,
    von_mises_feature_score,
)


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _interior_angles(v0, v1, v2):
    a = _dist(v1, v2)
    b = _dist(v0, v2)
    c = _dist(v0, v1)
    if a * b == 0 or b * c == 0 or c * a == 0:
        return [0.0, 0.0, 0.0]
    try:
        angle_a = math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c)))))
        angle_b = math.degrees(math.acos(max(-1.0, min(1.0, (a * a + c * c - b * b) / (2 * a * c)))))
        angle_c = 180.0 - angle_a - angle_b
    except ValueError:
        return [0.0, 0.0, 0.0]
    return sorted([angle_a, angle_b, angle_c])


def legacy_extract_features_from_row(
    row,
    include_shape_features: bool = False,
    include_side_ratios: bool = True,
    include_radius_ratio: bool = True,
    include_elongation: bool = True,
    eps: float = 1e-8,
) -> List[float]:
    """Original feature extractor; supports optional shape descriptors.

    The paper's 5-D descriptor is ``include_shape_features=False`` and yields
    ``[alpha1, alpha2, e1, e2, e3]``. The other flags are kept for backward
    compatibility but are not part of the paper specification.
    """
    v0 = (row["vertex_0_x"], row["vertex_0_y"])
    v1 = (row["vertex_1_x"], row["vertex_1_y"])
    v2 = (row["vertex_2_x"], row["vertex_2_y"])

    geo_angles = _interior_angles(v0, v1, v2)
    eke_angles = sorted([row["ekeland_v0"], row["ekeland_v1"], row["ekeland_v2"]])
    features = geo_angles[:2] + eke_angles

    if not include_shape_features:
        return features

    a = _dist(v1, v2)
    b = _dist(v0, v2)
    c = _dist(v0, v1)
    s0, s1, s2 = sorted([a, b, c])
    if include_side_ratios:
        features.extend([s0 / (s1 + eps), s1 / (s2 + eps), s0 / (s2 + eps)])
    if include_radius_ratio:
        semi = 0.5 * (a + b + c)
        area_sq = max(semi * (semi - a) * (semi - b) * (semi - c), 0.0)
        area = math.sqrt(area_sq)
        denom = max(semi * a * b * c, eps)
        features.append((4.0 * area * area) / denom)
    if include_elongation:
        features.append(s2 / (s0 + eps))
    return features


def legacy_brute_force_matching(
    df_uav: pd.DataFrame,
    df_sat: pd.DataFrame,
    top_k: int = 1000,
    metric: str = "l1",
    vm_kappa: float = 4.0,
    angle_feature_dims: int = 5,
    linear_weight: float = 1.0,
):
    """Old O(M*N) matching: returns top-K (uav_idx, sat_idx) pairs by ascending score."""
    metric = metric.lower()
    uav_features = [legacy_extract_features_from_row(row) for _, row in df_uav.iterrows()]
    sat_features = [legacy_extract_features_from_row(row) for _, row in df_sat.iterrows()]

    def _score(feat_a, feat_b):
        angle_a, angle_b = feat_a[:angle_feature_dims], feat_b[:angle_feature_dims]
        linear_a, linear_b = feat_a[angle_feature_dims:], feat_b[angle_feature_dims:]
        if metric == "l1":
            return l1_distance(feat_a, feat_b)
        if metric == "l2":
            return l2_distance(feat_a, feat_b)
        if metric == "complex":
            base = complex_angle_distance(angle_a, angle_b)
        elif metric == "geodesic":
            base = geodesic_feature_distance(angle_a, angle_b)
        elif metric == "circular_mean":
            base = circular_mean_distance(angle_a, angle_b)
        elif metric == "von_mises":
            base = von_mises_feature_score(angle_a, angle_b, kappa=vm_kappa)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if linear_a and linear_b:
            base += linear_weight * l1_distance(linear_a, linear_b)
        return base

    top_heap: list = []
    for idx_u, feat_u in enumerate(uav_features):
        for idx_s, feat_s in enumerate(sat_features):
            s = _score(feat_u, feat_s)
            candidate = (-s, idx_u, idx_s)
            if len(top_heap) < top_k:
                heapq.heappush(top_heap, candidate)
            elif s < -top_heap[0][0]:
                heapq.heapreplace(top_heap, candidate)

    top_candidates = sorted(top_heap, key=lambda x: -x[0])
    return [
        {
            "score": -neg,
            "uav_idx": idx_u,
            "sat_idx": idx_s,
            "uav_row": df_uav.iloc[idx_u],
            "sat_row": df_sat.iloc[idx_s],
        }
        for neg, idx_u, idx_s in top_candidates
    ]
