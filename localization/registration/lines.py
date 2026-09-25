"""Training-free structural frontend: man-made line structure (paper §4.2, 'Lines').

No learned component. Straight line segments (LSD, von Gioi et al.) are
extracted at the common metric resolution; only segments at least
``min_len_m`` long are kept (building outlines, road edges, plot walls --
man-made structure; short texture edges are discarded). Uncertainty of the
extraction is propagated exactly as for the segmentation frontend: a
structural ensemble over image scale and pre-smoothing yields, for every
boundary sample, the fraction of members that reproduce a line within r
(persistence p_b), which weights the evidence.

The output plugs into the same objective: the reference kernel map G_S is
the smoothed chamfer of the line raster; the query is its weighted boundary
point set. There is no region (mask) term in this frontend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .structure import QueryStructure, RefMaps


@dataclass
class LineConfig:
    min_len_m: float = 6.0
    scales: Sequence[float] = (0.75, 1.0, 1.33)      # ensemble: image rescale factors
    blurs_px: Sequence[float] = (0.0, 1.0)           # ensemble: pre-smoothing sigma
    match_radius_m: float = 1.5
    tile: int = 4096


def _lsd(gray: np.ndarray):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    out = lsd.detect(gray)[0]
    return np.zeros((0, 4), np.float32) if out is None else out.reshape(-1, 4)


def detect_segments(gray: np.ndarray, gsd: float, min_len_m: float, scale=1.0, blur=0.0, tile=4096):
    """Segments (x1,y1,x2,y2) in the input pixel frame; large rasters are tiled."""
    g = gray
    if blur > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    H, W = g.shape
    ov = 64
    segs = []
    for y in range(0, H, tile - ov):
        for x in range(0, W, tile - ov):
            s = _lsd(g[y:y + tile, x:x + tile])
            if len(s):
                s = s + np.array([x, y, x, y], np.float32)
                segs.append(s)
    s = np.concatenate(segs) if segs else np.zeros((0, 4), np.float32)
    s = s / scale
    L = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) * gsd
    return s[L >= min_len_m]


def segments_m(gray: np.ndarray, gsd: float, cfg: "LineConfig | None" = None, centre: bool = False) -> np.ndarray:
    """Line segments in metres (x east, y south); origin = image centre if ``centre``."""
    cfg = cfg or LineConfig()
    s = detect_segments(gray, gsd, cfg.min_len_m * 0.5, tile=cfg.tile).astype(np.float64)
    if centre:
        h, w = gray.shape
        s = s - np.array([(w - 1) / 2, (h - 1) / 2, (w - 1) / 2, (h - 1) / 2])
    return s * gsd


def rasterize(segs: np.ndarray, shape, thickness=1) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in segs:
        cv2.line(m, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 1, thickness)
    return m


def line_maps(gray: np.ndarray, gsd: float, gsd_out: float, cfg: LineConfig | None = None,
              valid: np.ndarray | None = None, persistence: bool = True):
    """Nominal line raster + per-pixel persistence, both at ``gsd_out``.

    Lines are detected at the input resolution ``gsd`` and rasterized directly
    at ``gsd_out`` (downsampling a 1-px raster would erase thin lines).
    ``valid`` (optional) is the footprint at ``gsd_out``.
    """
    cfg = cfg or LineConfig()
    k = gsd / gsd_out
    shape = (int(round(gray.shape[0] * k)), int(round(gray.shape[1] * k)))
    nominal = rasterize(detect_segments(gray, gsd, cfg.min_len_m, tile=cfg.tile) * k, shape)
    r_px = cfg.match_radius_m / gsd_out
    acc = np.zeros(shape, np.float32); n = 0
    for sc in (cfg.scales if persistence else ()):
        for bl in cfg.blurs_px:
            m = rasterize(detect_segments(gray, gsd, cfg.min_len_m, sc, bl, cfg.tile) * k, shape)
            d = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
            acc += (d <= r_px); n += 1
    pers = acc / n if n else np.ones(shape, np.float32)
    if valid is not None:
        # drop lines on the footprint border (image edge / rotation padding artefacts)
        v = cv2.resize(valid.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        inner = cv2.erode(v, np.ones((7, 7), np.uint8))
        nominal = nominal * inner
    return nominal, pers


def reference_from_lines(nominal: np.ndarray, gsd: float, sigma_m: float = 2.0) -> RefMaps:
    d = cv2.distanceTransform((1 - (nominal > 0)).astype(np.uint8), cv2.DIST_L2, 5) * gsd
    G = np.exp(-(d * d) / (2.0 * sigma_m * sigma_m)).astype(np.float32)
    return RefMaps(G=G, M=np.zeros_like(G), gsd=gsd)


def query_from_lines(nominal: np.ndarray, pers: np.ndarray, gsd: float, valid: np.ndarray,
                     use_persistence: bool = True, max_points: int = 20000, seed: int = 0) -> QueryStructure:
    ys, xs = np.nonzero(nominal)
    w = pers[ys, xs].astype(np.float32) if use_persistence else np.ones(len(xs), np.float32)
    if len(xs) > max_points:
        sel = np.random.default_rng(seed).choice(len(xs), max_points, replace=False)
        xs, ys, w = xs[sel], ys[sel], w[sel]
    h, wd = nominal.shape
    cx, cy = (wd - 1) / 2.0, (h - 1) / 2.0
    pts = np.stack([(xs - cx) * gsd, (ys - cy) * gsd], 1).astype(np.float32)
    # number of 'structures' ~ connected line groups (for integrity/strata)
    n_cc = cv2.connectedComponents(cv2.dilate(nominal, np.ones((3, 3), np.uint8)))[0] - 1
    return QueryStructure(pts=pts, w=w, mask=np.zeros(nominal.shape, np.float32), valid=valid.astype(np.uint8),
                          gsd=gsd, n_buildings=int(n_cc), polygons=[],
                          mean_persistence=float(w.mean()) if len(w) else 0.0)
