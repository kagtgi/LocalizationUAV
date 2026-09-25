"""Ekeland / MFCA building-shape verification of a pose hypothesis (§4.6).

For a hypothesis T every UAV building is paired with the satellite building
whose centroid is nearest to T(c_i). Agreement

    A(T) = (1/|B_U|) sum_i  exp(-||f_i - f_j(i)||_1 / sigma_f) * 1[d_ij < r]

uses a building-level signature built from *vertex-coupled* MFCA triangle
descriptors: the vertices of each CDT triangle are ordered once by interior
angle and the Ekeland angle of each vertex keeps its position, so each
vertex's (angle, free-cone angle) pair survives (plan Stage E, M2).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from ..geometry.descriptor import _interior_simplex_mask
from ..geometry.ekeland import compute_expansion_ekeland_for_all_triangles


def _angles(P):
    a, b, c = P
    def ang(p, q, r):
        v1, v2 = q - p, r - p
        cs = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        return math.degrees(math.acos(max(-1.0, min(1.0, cs))))
    return np.array([ang(a, b, c), ang(b, c, a), ang(c, a, b)])


def coupled_descriptors(poly: np.ndarray, max_depth: int = 4, coupled: bool = True) -> np.ndarray:
    """(T,5) triangle descriptors; ``coupled=False`` reproduces M0's independent sort."""
    P = np.asarray(poly, np.float64)
    if len(P) < 3:
        return np.zeros((0, 5), np.float32)
    try:
        tri = Delaunay(P)
    except Exception:
        return np.zeros((0, 5), np.float32)
    mask = _interior_simplex_mask(tri, P)
    res = compute_expansion_ekeland_for_all_triangles(tri, interior_mask=mask, max_depth=max_depth)
    out = []
    for r in res:
        V = np.asarray(r["seed_coordinates"], np.float64)
        ids = r["seed_vertices"]
        e = np.array([r["ekeland_angles"].get(v, 0.0) for v in ids])
        a = _angles(V)
        if coupled:
            # deterministic vertex order: interior angle, tie-break by adjacent-edge ratio
            L = np.array([np.linalg.norm(V[(i + 1) % 3] - V[(i + 2) % 3]) for i in range(3)])
            key = [(a[i], L[(i + 1) % 3] / (L[(i + 2) % 3] + 1e-9)) for i in range(3)]
            pi = sorted(range(3), key=lambda i: key[i])
            out.append([a[pi[0]], a[pi[1]], e[pi[0]], e[pi[1]], e[pi[2]]])
        else:
            aa = np.sort(a); ee = np.sort(e)
            out.append([aa[0], aa[1], ee[0], ee[1], ee[2]])
    return np.asarray(out, np.float32) if out else np.zeros((0, 5), np.float32)


def building_signature(poly: np.ndarray, max_depth: int = 4, coupled: bool = True) -> np.ndarray:
    """Rotation/scale-invariant 7-D building signature from MFCA triangles."""
    d = coupled_descriptors(poly, max_depth, coupled)
    if len(d) == 0:
        return np.array([0.5, 0.5, 1, 1, 1, 1, 0], np.float32)
    d = d / 180.0
    return np.concatenate([d.mean(0), [d[:, 2:].min(), math.log1p(len(d)) / 3.0]]).astype(np.float32)


def signatures(polys: Sequence[np.ndarray], max_depth: int = 4, coupled: bool = True, workers: int = 0) -> np.ndarray:
    if workers and len(polys) > 200:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_sig_star, [(p, max_depth, coupled) for p in polys], chunksize=64))
    else:
        out = [building_signature(p, max_depth, coupled) for p in polys]
    return np.asarray(out, np.float32).reshape(-1, 7)


def _sig_star(a):
    return building_signature(*a)


class ShapeVerifier:
    """Pre-indexed satellite buildings (reference pixel coordinates)."""

    def __init__(self, sat_cents_px: np.ndarray, sat_sigs: np.ndarray, sat_area_px: np.ndarray, gsd: float):
        self.tree = cKDTree(sat_cents_px) if len(sat_cents_px) else None
        self.sigs = sat_sigs
        self.area = sat_area_px
        self.gsd = gsd

    def agreement(self, q_polys_m: List[np.ndarray], q_sigs: np.ndarray, u: float, v: float,
                  theta_deg: float, s: float, r_m: float = 8.0, sigma_f: float = 0.15) -> dict:
        if self.tree is None or not q_polys_m:
            return {"A": 0.0, "matched": 0, "n": len(q_polys_m)}
        th = math.radians(theta_deg)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        cents = np.array([p.mean(0) for p in q_polys_m])
        areas = np.array([_area(p) for p in q_polys_m])
        cp = (cents @ R.T) * (s / self.gsd) + np.array([u, v])
        d, j = self.tree.query(cp, k=1)
        ok = d * self.gsd < r_m
        # size agreement (log-area ratio) as a guard against matching a shed to a block
        qa = areas * (s * s) / (self.gsd ** 2)
        size_ok = np.abs(np.log((qa + 1) / (self.area[j] + 1))) < math.log(2.5)
        ok = ok & size_ok
        sim = np.exp(-np.abs(q_sigs - self.sigs[j]).sum(1) / sigma_f)
        A = float((sim * ok).sum() / max(len(q_polys_m), 1))
        return {"A": A, "matched": int(ok.sum()), "n": len(q_polys_m)}


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_areas(polys):
    return np.array([_area(np.asarray(p, np.float64)) for p in polys], np.float32)
