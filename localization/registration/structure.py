"""Canonical structural maps (paper §4.1-4.3).

All maps live on a metric raster of ground sampling distance ``gsd`` (m/px).
The query side is a *point set* (boundary samples with persistence weights)
plus a signed mask; the reference side is a smoothed-chamfer kernel map
``G_S`` and a signed mask ``M_S``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# resampling
# ----------------------------------------------------------------------------


def resample(arr: np.ndarray, src_gsd: float, dst_gsd: float) -> np.ndarray:
    """Resample a raster from ``src_gsd`` to ``dst_gsd`` (m/px)."""
    f = float(src_gsd) / float(dst_gsd)
    if abs(f - 1.0) < 1e-6:
        return arr
    h, w = arr.shape[:2]
    nh, nw = max(1, int(round(h * f))), max(1, int(round(w * f)))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    return cv2.resize(arr, (nw, nh), interpolation=interp)


def boundary_of(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary of a binary mask (uint8 0/1)."""
    m = (mask > 0).astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    return (m - er).astype(np.uint8)


def remove_small(mask: np.ndarray, min_area_px: float) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[lab].astype(np.uint8)


def simplify_mask(mask: np.ndarray, tol_px: float) -> np.ndarray:
    """Re-rasterize a mask after Douglas-Peucker simplification of its contours."""
    if tol_px <= 0:
        return mask
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask, dtype=np.uint8)
    polys = [cv2.approxPolyDP(c, tol_px, True) for c in cs]
    polys = [p for p in polys if len(p) >= 3]
    if polys:
        cv2.fillPoly(out, polys, 1)
    return out


# ----------------------------------------------------------------------------
# structural ensemble -> persistence (uncertainty propagation, §4.2)
# ----------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7)
    dp_tol_m: Sequence[float] = (0.5, 1.0, 2.0)
    match_radius_m: float = 1.5
    min_area_m2: float = 20.0


def ensemble_masks(prob: np.ndarray, gsd: float, cfg: EnsembleConfig):
    """Structural ensemble: masks over segmentation thresholds x DP tolerances."""
    min_px = cfg.min_area_m2 / (gsd * gsd)
    out = []
    for th in cfg.thresholds:
        m = remove_small(prob > th, min_px)
        for tol in cfg.dp_tol_m:
            out.append(simplify_mask(m, tol / gsd))
    return out


def persistence_map(prob: np.ndarray, gsd: float, cfg: EnsembleConfig) -> np.ndarray:
    """p(x) = fraction of ensemble members with a boundary within r of x.

    Evaluated densely; sampled at the nominal boundary it is the persistence
    ``p_b`` of each boundary point (plan Stage C/D, raster form).
    """
    r_px = cfg.match_radius_m / gsd
    acc = np.zeros(prob.shape, np.float32)
    members = ensemble_masks(prob, gsd, cfg)
    for m in members:
        b = boundary_of(m)
        d = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2, 3)
        acc += (d <= r_px).astype(np.float32)
    return acc / max(len(members), 1)


# ----------------------------------------------------------------------------
# query structure: weighted boundary points + signed mask
# ----------------------------------------------------------------------------


@dataclass
class QueryStructure:
    pts: np.ndarray        # (N,2) boundary points, metres, centred on image centre, x=east, y=south
    w: np.ndarray          # (N,) weights p_b * c_b
    mask: np.ndarray       # (h,w) float32 signed mask in [-1,1] (0 outside footprint)
    valid: np.ndarray      # (h,w) uint8 footprint support
    gsd: float
    n_buildings: int
    polygons: list         # list of (K,2) arrays in metres (same frame as pts)
    mean_persistence: float

    @property
    def coverage(self) -> float:
        """Fraction of the observed footprint covered by buildings (no GT used)."""
        v = self.valid > 0
        return float((self.mask[v] > 0).mean()) if v.any() else 0.0


def query_structure(
    prob: np.ndarray,
    gsd: float,
    valid: np.ndarray | None = None,
    ens: EnsembleConfig | None = None,
    use_persistence: bool = True,
    max_points: int = 20000,
    rng: np.random.Generator | None = None,
) -> QueryStructure:
    """Build the query structure from a building-probability map at ``gsd``."""
    ens = ens or EnsembleConfig()
    rng = rng or np.random.default_rng(0)
    h, w = prob.shape
    if valid is None:
        valid = np.ones((h, w), np.uint8)
    min_px = ens.min_area_m2 / (gsd * gsd)
    m = remove_small((prob > 0.5) & (valid > 0), min_px)
    b = boundary_of(m) & (cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0)
    ys, xs = np.nonzero(b)
    if use_persistence:
        pers = persistence_map(prob * (valid > 0), gsd, ens)
        pb = pers[ys, xs]
    else:
        pb = np.ones(len(xs), np.float32)
    # segmentation confidence: distance of prob from 0.5 in a small band
    cb = np.clip(np.abs(cv2.GaussianBlur(prob, (0, 0), 1.0)[ys, xs] - 0.5) * 2 + 0.5, 0.5, 1.0) if use_persistence else np.ones(len(xs), np.float32)
    wts = (pb * cb).astype(np.float32)
    if len(xs) > max_points:
        sel = rng.choice(len(xs), max_points, replace=False)
        xs, ys, wts = xs[sel], ys[sel], wts[sel]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    pts = np.stack([(xs - cx) * gsd, (ys - cy) * gsd], 1).astype(np.float32)

    signed = np.where(valid > 0, 2.0 * m.astype(np.float32) - 1.0, 0.0).astype(np.float32)

    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in cs:
        p = cv2.approxPolyDP(c, 1.0 / gsd, True).reshape(-1, 2).astype(np.float32)
        if len(p) >= 3:
            polys.append(np.stack([(p[:, 0] - cx) * gsd, (p[:, 1] - cy) * gsd], 1))
    return QueryStructure(
        pts=pts, w=wts, mask=signed, valid=valid.astype(np.uint8), gsd=gsd,
        n_buildings=len(polys), polygons=polys,
        mean_persistence=float(pb.mean()) if len(pb) else 0.0,
    )


# ----------------------------------------------------------------------------
# reference structure: kernel map + signed mask
# ----------------------------------------------------------------------------


@dataclass
class RefMaps:
    G: np.ndarray      # smoothed chamfer kernel exp(-D^2 / 2 sigma^2), float32
    M: np.ndarray      # signed mask 2m-1, float32
    gsd: float


def reference_maps(prob: np.ndarray, gsd: float, sigma_m: float = 2.0, min_area_m2: float = 20.0) -> RefMaps:
    m = remove_small(prob > 0.5, min_area_m2 / (gsd * gsd))
    b = boundary_of(m)
    d = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2, 5) * gsd
    G = np.exp(-(d * d) / (2.0 * sigma_m * sigma_m)).astype(np.float32)
    M = (2.0 * m.astype(np.float32) - 1.0).astype(np.float32)
    return RefMaps(G=G, M=M, gsd=gsd)


def reference_polygons(prob: np.ndarray, gsd: float, min_area_m2: float = 20.0):
    """Satellite building polygons (pixel coords at ``gsd``) + centroids, for verification."""
    m = remove_small(prob > 0.5, min_area_m2 / (gsd * gsd))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys, cents = [], []
    for c in cs:
        p = cv2.approxPolyDP(c, 1.0 / gsd, True).reshape(-1, 2).astype(np.float32)
        if len(p) >= 3:
            polys.append(p)
            cents.append(p.mean(0))
    return polys, (np.asarray(cents, np.float32) if cents else np.zeros((0, 2), np.float32))
