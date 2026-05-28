"""Extra distance metrics that are NOT part of the paper.

The paper §4.5 specifies a single distance metric: ``ell_1`` (cityblock).
These additional metrics existed in the original ``matching/features.py`` for
exploratory experiments and are preserved here for reference / ablation only.
They are *not* used by the production pipeline in ``localization/``.
"""

from __future__ import annotations

import math

import numpy as np


def l1_distance(feat_a, feat_b):
    return sum(abs(a - b) for a, b in zip(feat_a, feat_b))


def l2_distance(feat_a, feat_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(feat_a, feat_b)))


def angular_distance(a: float, b: float) -> float:
    """Shortest angular distance between two angles in degrees, in [0, 180]."""
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff


def complex_angle_distance(feat_a, feat_b):
    """Chord distance between unit-circle embeddings of each angular coordinate."""
    s = 0.0
    for a, b in zip(feat_a, feat_b):
        rad_a, rad_b = np.radians(a), np.radians(b)
        s += np.sqrt(2 * (1 - np.cos(rad_a - rad_b)))
    return float(s)


def geodesic_feature_distance(feat_a, feat_b):
    """Sum of shortest angular distances per dimension."""
    return float(sum(angular_distance(a, b) for a, b in zip(feat_a, feat_b)))


def circular_mean_distance(feat_a, feat_b):
    """Mean of ``1 - cos(delta)`` per dimension."""
    vals = []
    for a, b in zip(feat_a, feat_b):
        delta_rad = np.radians(angular_distance(a, b))
        vals.append(1.0 - np.cos(delta_rad))
    return float(np.mean(vals))


def von_mises_feature_score(feat_a, feat_b, kappa: float = 4.0):
    """Sum of per-dimension von Mises NLL (full normalization kept)."""
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    nll = 0.0
    for a, b in zip(feat_a, feat_b):
        delta_rad = np.radians(angular_distance(a, b))
        nll += (-kappa * np.cos(delta_rad)) + np.log(2.0 * np.pi * np.i0(kappa))
    return float(nll)
