"""Step 3 - Constrained Delaunay Triangulation + Phase 1 kernel expansion.

Paper §4.3, Algorithm 1 Phase 1: build P_sub by merging adjacent CDT triangles
while keeping all 3 original triangle vertices on the boundary (Ekeland
criticality condition).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


class TriangulationGraph:
    """Dual graph + region expansion over a Delaunay triangulation.

    An optional ``valid_mask`` (bool array over ``triangulation.simplices``)
    restricts the dual graph to *interior* triangles only: exterior triangles
    (those a convex-hull Delaunay triangulation places outside a non-convex
    footprint) are never connected, so kernel expansion stays inside the
    polygon boundary - the boundary-respecting (constrained) behaviour required
    by paper §4.3.
    """

    def __init__(self, triangulation, valid_mask=None):
        self.triangulation = triangulation
        self.valid_mask = valid_mask

    def _is_valid(self, tri_idx: int) -> bool:
        return self.valid_mask is None or bool(self.valid_mask[tri_idx])

    def build_dual_graph(self) -> Dict[int, List[int]]:
        n = len(self.triangulation.simplices)
        dual: Dict[int, List[int]] = {i: [] for i in range(n)}
        for tri_idx in range(n):
            if not self._is_valid(tri_idx):
                continue
            for j in range(3):
                neighbor_idx = int(self.triangulation.neighbors[tri_idx, j])
                if neighbor_idx != -1 and self._is_valid(neighbor_idx):
                    dual[tri_idx].append(neighbor_idx)
        return dual

    def get_triangle_vertices(self, triangle_idx: int):
        simplex = self.triangulation.simplices[triangle_idx]
        vertices = self.triangulation.points[simplex]
        return simplex, vertices

    def get_region_boundary_edges(self, region_triangle_indices) -> List[Tuple[int, int]]:
        edge_count: Dict[Tuple[int, int], int] = {}
        for tri_idx in region_triangle_indices:
            simplex = self.triangulation.simplices[tri_idx]
            edges = [
                tuple(sorted([int(simplex[0]), int(simplex[1])])),
                tuple(sorted([int(simplex[1]), int(simplex[2])])),
                tuple(sorted([int(simplex[2]), int(simplex[0])])),
            ]
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        return [edge for edge, count in edge_count.items() if count == 1]

    def get_region_boundary_vertices(self, region_triangle_indices) -> Set[int]:
        vertices: Set[int] = set()
        for v1, v2 in self.get_region_boundary_edges(region_triangle_indices):
            vertices.add(v1)
            vertices.add(v2)
        return vertices

    def order_boundary_vertices(self, boundary_edges: List[Tuple[int, int]]) -> List[int]:
        """Order boundary vertices into a single closed loop."""
        if not boundary_edges:
            return []

        adjacency: Dict[int, List[int]] = {}
        for v1, v2 in boundary_edges:
            adjacency.setdefault(v1, []).append(v2)
            adjacency.setdefault(v2, []).append(v1)

        start_vertex = boundary_edges[0][0]
        ordered = [start_vertex]
        visited = {start_vertex}
        current = start_vertex
        while len(ordered) < len(adjacency):
            next_vertex = None
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    next_vertex = neighbor
                    break
            if next_vertex is None:
                break
            ordered.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex
        return ordered

    def _shared_edge_length(self, region_triangle_indices, candidate_tri_idx: int) -> float:
        """Length of the edge(s) ``candidate_tri_idx`` shares with the current
        region boundary - a geometric, serialization-invariant merge priority
        (see :meth:`expand_from_seed_triangle`).
        """
        boundary_edges = set(self.get_region_boundary_edges(region_triangle_indices))
        simplex = self.triangulation.simplices[candidate_tri_idx]
        candidate_edges = [
            tuple(sorted([int(simplex[0]), int(simplex[1])])),
            tuple(sorted([int(simplex[1]), int(simplex[2])])),
            tuple(sorted([int(simplex[2]), int(simplex[0])])),
        ]
        shared = [e for e in candidate_edges if e in boundary_edges]
        if not shared:
            return 0.0
        pts = self.triangulation.points
        return float(sum(np.linalg.norm(pts[e[0]] - pts[e[1]]) for e in shared))

    def expand_from_seed_triangle(self, dual_graph, seed_triangle_idx, max_iterations: int = 100) -> dict:
        """Phase 1 - merge adjacent triangles into P_sub while keeping all three
        seed vertices on the boundary (Ekeland criticality stopping condition).

        Among all valid candidates in the frontier, the one sharing the
        *longest* edge with the current region boundary is merged first. This
        is a purely geometric priority rule: it depends only on vertex
        coordinates, not on how the CDT happens to index its triangles, so
        the resulting P_sub is invariant to triangle serialization order
        (matching triangulations that differ only in index assignment, e.g.
        from minor vertex-ordering noise between a UAV mask and its satellite
        counterpart, still expand identically). Triangle index is used only
        as a last-resort tiebreak for exact ties in shared-edge length.
        """
        original_simplex, _ = self.get_triangle_vertices(seed_triangle_idx)
        original_vertices = set(int(v) for v in original_simplex)

        current_region: Set[int] = {seed_triangle_idx}
        frontier: Set[int] = set(dual_graph[seed_triangle_idx])
        visited: Set[int] = set(dual_graph[seed_triangle_idx]) | {seed_triangle_idx}

        expansion_steps = 0
        for _ in range(max_iterations):
            best_candidate = None
            best_score = None
            for candidate_tri_idx in frontier:
                temp_region = current_region | {candidate_tri_idx}
                temp_boundary_edges = self.get_region_boundary_edges(temp_region)
                temp_boundary_vertices = self.get_region_boundary_vertices(temp_region)
                original_still_on_boundary = original_vertices.issubset(temp_boundary_vertices)
                ordered_loop = self.order_boundary_vertices(temp_boundary_edges)
                no_holes = len(ordered_loop) == len(temp_boundary_vertices)
                if not (original_still_on_boundary and no_holes):
                    continue

                shared_len = self._shared_edge_length(current_region, candidate_tri_idx)
                score = (shared_len, -candidate_tri_idx)
                if best_score is None or score > best_score:
                    best_score = score
                    best_candidate = candidate_tri_idx

            if best_candidate is None:
                break

            current_region = current_region | {best_candidate}
            frontier.remove(best_candidate)
            for neighbor_idx in dual_graph[best_candidate]:
                if neighbor_idx not in visited:
                    frontier.add(neighbor_idx)
                    visited.add(neighbor_idx)
            expansion_steps += 1

        final_boundary_edges = self.get_region_boundary_edges(current_region)
        current_boundary_vertices = self.get_region_boundary_vertices(current_region)
        return {
            "region_triangles": current_region,
            "boundary_vertices": current_boundary_vertices,
            "boundary_edges": final_boundary_edges,
            "original_vertices": original_vertices,
            "expansion_steps": expansion_steps,
            "seed_triangle": seed_triangle_idx,
        }


# ---------------- Functional API used by the rest of the package -----------------


def build_dual_graph(triangulation, valid_mask=None) -> Dict[int, List[int]]:
    return TriangulationGraph(triangulation, valid_mask=valid_mask).build_dual_graph()


def get_triangle_vertices(triangulation, triangle_idx: int):
    return TriangulationGraph(triangulation).get_triangle_vertices(triangle_idx)


def get_region_boundary_edges(triangulation, region_triangle_indices):
    return TriangulationGraph(triangulation).get_region_boundary_edges(region_triangle_indices)


def get_region_boundary_vertices(triangulation, region_triangle_indices):
    return TriangulationGraph(triangulation).get_region_boundary_vertices(region_triangle_indices)


def order_boundary_vertices(triangulation, boundary_edges):
    return TriangulationGraph(triangulation).order_boundary_vertices(boundary_edges)


def expand_from_seed_triangle(triangulation, dual_graph, seed_triangle_idx, max_iterations: int = 100):
    return TriangulationGraph(triangulation).expand_from_seed_triangle(
        dual_graph=dual_graph,
        seed_triangle_idx=seed_triangle_idx,
        max_iterations=max_iterations,
    )
