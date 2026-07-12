#!/usr/bin/env python
"""Search for 3 concrete, real-algorithm-verified fig_descriptor examples
spanning the Ekeland angle's actual range:
  case 1: e_v = 180 (convex, exact)
  case 2: 90 < e_v < 180 (shallow/obtuse concavity)
  case 3: e_v < 90 (sharp/deep concavity)
All values computed by the real project code -- nothing hand-derived.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay

from localization.geometry.descriptor import _interior_simplex_mask
from localization.geometry.triangulation import build_dual_graph, expand_from_seed_triangle
from localization.geometry.ekeland import calculate_ekeland_angles_on_boundary


def sweep(poly, label, lo=None, hi=None):
    poly = np.asarray(poly, float)
    tri = Delaunay(poly)
    mask = _interior_simplex_mask(tri, poly)
    dual = build_dual_graph(tri, valid_mask=mask)
    hits = []
    for seed_idx in range(len(tri.simplices)):
        if not mask[seed_idx]:
            continue
        for d in [0, 1, 2, 3, 4]:
            exp = expand_from_seed_triangle(tri, dual, seed_idx, max_iterations=d)
            e = calculate_ekeland_angles_on_boundary(tri, exp)
            for v in tri.simplices[seed_idx]:
                v = int(v)
                ev = e[v]["ekeland_angle"]
                if lo is not None and not (lo <= ev <= hi):
                    continue
                hits.append((seed_idx, v, d, round(ev, 2), sorted(exp["region_triangles"])))
    print(f"=== {label}: {int(mask.sum())} interior tris, {len(hits)} hits in range ===")
    for h in hits[:10]:
        print(f"  seed={h[0]} vertex={h[1]} coords={poly[h[1]]} D_max={h[2]} e_v={h[3]} region={h[4]}")
    return poly, tri, mask, dual, hits


if __name__ == "__main__":
    # shallow/obtuse V-notch: wide opening, shallow depth -> mild reflex angle
    shallow_v = [(0, 0), (3, 0), (3, 3.2), (2.3, 3.2), (1.5, 2.4), (0.7, 3.2), (0, 3.2)]
    sweep(shallow_v, "shallow-V-notch (target 90<e_v<180)", 90.001, 179.999)

    # sharp/acute V-notch: narrow opening, deep -> strong reflex angle
    sharp_v = [(0, 0), (3, 0), (3, 3.2), (1.65, 3.2), (1.5, 0.9), (1.35, 3.2), (0, 3.2)]
    sweep(sharp_v, "sharp-V-notch (target e_v<90)", 0.001, 89.999)
