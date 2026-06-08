"""Step 4 - locked 5-D triangle descriptor (paper §4.4 Eq. f).

    f = (alpha_1, alpha_2, e_1, e_2, e_3)

where alpha_1 <= alpha_2 are the two smallest sorted interior angles, and
e_1 <= e_2 <= e_3 are the three sorted Ekeland angles - all in (0, 180) deg.

No optional shape, side-ratio, radius, or elongation flags: any deviation
from this 5-tuple does not match the paper's specification.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon

from .ekeland import compute_expansion_ekeland_for_all_triangles


VertexXY = Tuple[float, float]


def _interior_simplex_mask(tri: "Delaunay", polygon_np: np.ndarray) -> np.ndarray:
    """Boolean mask over Delaunay simplices: True iff the triangle is inside the polygon.

    Implements the boundary-respecting (Constrained) Delaunay Triangulation of
    paper §4.3 by *interior filtering*: an unconstrained convex-hull Delaunay
    triangulation emits triangles outside a non-convex footprint; we keep only
    those whose centroid lies inside the (repaired) polygon. For simple building
    footprints this reproduces the interior mesh of a boundary-edge-constrained
    Delaunay triangulation without an extra geometry dependency.
    """
    n = len(tri.simplices)
    try:
        poly = Polygon(polygon_np)
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return np.ones(n, dtype=bool)
    if poly.is_empty or poly.area <= 0.0:
        return np.ones(n, dtype=bool)

    centroids = polygon_np[tri.simplices].mean(axis=1)  # (T, 2)
    mask = np.fromiter(
        (poly.contains(Point(float(cx), float(cy))) for cx, cy in centroids),
        dtype=bool,
        count=n,
    )
    # Degenerate guard: if nothing survived, fall back to the full set.
    return mask if mask.any() else np.ones(n, dtype=bool)


def _dist(p1: VertexXY, p2: VertexXY) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def interior_angles(v0: VertexXY, v1: VertexXY, v2: VertexXY) -> List[float]:
    """Three interior angles of a triangle (deg), sorted ascending."""
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


def triangle_descriptor(
    v0: VertexXY,
    v1: VertexXY,
    v2: VertexXY,
    ekeland_v0: float,
    ekeland_v1: float,
    ekeland_v2: float,
) -> np.ndarray:
    """Assemble the 5-D descriptor (alpha_1, alpha_2, e_1, e_2, e_3)."""
    angles_sorted = interior_angles(v0, v1, v2)
    alpha1, alpha2 = angles_sorted[0], angles_sorted[1]
    eke_sorted = sorted([float(ekeland_v0), float(ekeland_v1), float(ekeland_v2)])
    return np.asarray([alpha1, alpha2, eke_sorted[0], eke_sorted[1], eke_sorted[2]], dtype=np.float32)


def triangle_descriptors_from_polygon(
    polygon_xy: Sequence[Sequence[float]], max_depth: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """CDT a polygon and return ``(descriptors (T,5), centroids (T,2))``.

    Vertices are the polygon's own vertices (no Steiner points). The triangulation
    respects the polygon boundary via interior filtering (see
    :func:`_interior_simplex_mask`), matching the Constrained Delaunay
    Triangulation of paper §4.3. ``max_depth`` is the paper's kernel-expansion
    cap ``D_max`` (Algorithm 1, default 4).
    """
    polygon_np = np.asarray(polygon_xy, dtype=np.float64)
    if polygon_np.shape[0] < 3:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    try:
        tri = Delaunay(polygon_np)
    except Exception:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    interior_mask = _interior_simplex_mask(tri, polygon_np)
    expansion_results = compute_expansion_ekeland_for_all_triangles(
        tri, interior_mask=interior_mask, max_depth=int(max_depth)
    )

    descriptors: List[np.ndarray] = []
    centroids: List[np.ndarray] = []
    for result in expansion_results:
        verts = result["seed_vertices"]
        coords = result["seed_coordinates"]
        eke = result["ekeland_angles"]
        v0, v1, v2 = (tuple(coords[0]), tuple(coords[1]), tuple(coords[2]))
        descriptor = triangle_descriptor(
            v0,
            v1,
            v2,
            ekeland_v0=eke.get(verts[0], 0.0),
            ekeland_v1=eke.get(verts[1], 0.0),
            ekeland_v2=eke.get(verts[2], 0.0),
        )
        descriptors.append(descriptor)
        centroids.append(
            np.array(
                [
                    (v0[0] + v1[0] + v2[0]) / 3.0,
                    (v0[1] + v1[1] + v2[1]) / 3.0,
                ],
                dtype=np.float32,
            )
        )

    if not descriptors:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    return np.vstack(descriptors), np.vstack(centroids)
