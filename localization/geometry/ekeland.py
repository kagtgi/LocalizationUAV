"""Step 4 - Ekeland (MFCA) angle per triangle vertex (paper §4.4).

For each seed triangle, build the kernel polygon P_sub via Phase 1 expansion
(triangulation.py), then compute the Ekeland angle at each of the three original
vertices via the closed-form equivalent of the axis-parallel reflection:

    e_X = min(360 deg - alpha_int(X), 180 deg)

This is the closed form of Definition 2 in the paper: the axis-parallel
reflection is an equivalent geometric reading (see the Remark after
Definition 2), but the value itself depends only on alpha_int(X), so this
O(1) formula is what is actually computed.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from shapely.geometry import Polygon

from .triangulation import (
    build_dual_graph,
    expand_from_seed_triangle,
    order_boundary_vertices,
)


def _outward_bisector(points, ordered_boundary, vertex_idx):
    """Outward bisector at ``vertex_idx`` on the polygon boundary."""
    coords = [points[v] for v in ordered_boundary]
    poly = Polygon(coords)
    if not poly.exterior.is_ccw:
        ordered_boundary = ordered_boundary[::-1]
        coords = coords[::-1]

    idx = ordered_boundary.index(vertex_idx)
    n = len(ordered_boundary)
    p_prev = np.asarray(coords[(idx - 1) % n])
    p_curr = np.asarray(coords[idx])
    p_next = np.asarray(coords[(idx + 1) % n])

    e1 = p_curr - p_prev
    e2 = p_next - p_curr
    n1 = np.array([e1[1], -e1[0]])
    n2 = np.array([e2[1], -e2[0]])
    n1 = n1 / (np.linalg.norm(n1) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2) + 1e-10)

    outward = n1 + n2
    norm = np.linalg.norm(outward)
    if norm < 1e-6:
        return n1
    return outward / norm


class EkelandAnalyzer:
    """Compute Ekeland angles for all triangles in a triangulation."""

    def calculate_ekeland_angles_on_boundary(self, triangulation, expansion_result) -> Dict[int, dict]:
        boundary_edges = expansion_result["boundary_edges"]
        original_vertices = expansion_result["original_vertices"]
        ordered_boundary = order_boundary_vertices(triangulation, boundary_edges)

        if len(ordered_boundary) < 3:
            return {
                v: {"ekeland_angle": 0.0, "best_axis": np.array([1, 0]), "axis_name": "Degenerate"}
                for v in original_vertices
            }

        results: Dict[int, dict] = {}
        for vertex_idx in original_vertices:
            if vertex_idx not in ordered_boundary:
                results[vertex_idx] = {
                    "ekeland_angle": 0.0,
                    "best_axis": np.array([1, 0]),
                    "axis_name": "Interior",
                }
                continue

            best_axis = _outward_bisector(triangulation.points, ordered_boundary, vertex_idx)

            coords = [triangulation.points[v] for v in ordered_boundary]
            poly = Polygon(coords)
            ordered = list(ordered_boundary)
            if not poly.exterior.is_ccw:
                ordered = ordered[::-1]
                coords = coords[::-1]
            idx = ordered.index(vertex_idx)
            n = len(ordered)
            p_prev = np.asarray(coords[(idx - 1) % n])
            p_curr = np.asarray(coords[idx])
            p_next = np.asarray(coords[(idx + 1) % n])

            v_prev = p_prev - p_curr
            v_next = p_next - p_curr
            angle_prev = np.degrees(np.arctan2(v_prev[1], v_prev[0]))
            angle_next = np.degrees(np.arctan2(v_next[1], v_next[0]))
            alpha_int = (angle_prev - angle_next) % 360.0
            alpha_ext = 360.0 - alpha_int
            ekeland_angle = float(min(alpha_ext, 180.0))

            results[vertex_idx] = {
                "ekeland_angle": ekeland_angle,
                "best_axis": best_axis,
                "axis_name": "Analytical",
            }
        return results

    def compute_expansion_ekeland_for_all_triangles(
        self, triangulation, interior_mask=None, max_depth: int = 4
    ) -> List[dict]:
        # ``interior_mask`` restricts both the dual graph and the seed set to
        # triangles inside the polygon (paper §4.3 boundary-respecting CDT).
        # ``max_depth`` is the paper's D_max kernel-expansion cap (Algorithm 1).
        dual = build_dual_graph(triangulation, valid_mask=interior_mask)
        results: List[dict] = []
        for seed_idx in range(len(triangulation.simplices)):
            if interior_mask is not None and not bool(interior_mask[seed_idx]):
                continue
            expansion = expand_from_seed_triangle(
                triangulation, dual, seed_idx, max_iterations=int(max_depth)
            )
            ekeland_results = self.calculate_ekeland_angles_on_boundary(triangulation, expansion)
            original_simplex = triangulation.simplices[seed_idx]
            results.append(
                {
                    "seed_triangle_idx": seed_idx,
                    "seed_vertices": original_simplex.tolist(),
                    "seed_coordinates": triangulation.points[original_simplex].tolist(),
                    "ekeland_angles": {v: ekeland_results[v]["ekeland_angle"] for v in ekeland_results},
                    "ekeland_full_results": ekeland_results,
                    "expansion_steps": expansion["expansion_steps"],
                    "num_triangles_in_region": len(expansion["region_triangles"]),
                    "num_boundary_vertices": len(expansion["boundary_vertices"]),
                }
            )
        return results


_DEFAULT_ANALYZER = EkelandAnalyzer()


def calculate_ekeland_angles_on_boundary(triangulation, expansion_result, **_kwargs):
    return _DEFAULT_ANALYZER.calculate_ekeland_angles_on_boundary(
        triangulation=triangulation, expansion_result=expansion_result
    )


def compute_expansion_ekeland_for_all_triangles(
    triangulation, interior_mask=None, max_depth: int = 4, **_kwargs
):
    return _DEFAULT_ANALYZER.compute_expansion_ekeland_for_all_triangles(
        triangulation=triangulation, interior_mask=interior_mask, max_depth=max_depth
    )
