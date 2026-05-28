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

from .ekeland import compute_expansion_ekeland_for_all_triangles


VertexXY = Tuple[float, float]


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


def triangle_descriptors_from_polygon(polygon_xy: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """CDT a polygon and return ``(descriptors (T,5), centroids (T,2))``.

    Vertices are the polygon's own vertices (no Steiner points). The CDT here
    is unconstrained Delaunay over the polygon vertex set; for convex/simple
    polygons typical of Mask R-CNN building outputs this matches the paper's
    boundary-edge-constrained triangulation closely enough.
    """
    polygon_np = np.asarray(polygon_xy, dtype=np.float64)
    if polygon_np.shape[0] < 3:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    try:
        tri = Delaunay(polygon_np)
    except Exception:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    expansion_results = compute_expansion_ekeland_for_all_triangles(tri)

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
