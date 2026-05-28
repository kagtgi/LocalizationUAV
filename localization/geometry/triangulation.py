"""Step 3 - Constrained Delaunay Triangulation + Phase 1 kernel expansion.

Paper §4.3, Algorithm 1 Phase 1: build P_sub by merging adjacent CDT triangles
while keeping all 3 original triangle vertices on the boundary (Ekeland
criticality condition).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


class TriangulationGraph:
    """Dual graph + region expansion over a Delaunay triangulation."""

    def __init__(self, triangulation):
        self.triangulation = triangulation

    def build_dual_graph(self) -> Dict[int, List[int]]:
        n = len(self.triangulation.simplices)
        dual: Dict[int, List[int]] = {i: [] for i in range(n)}
        for tri_idx in range(n):
            for j in range(3):
                neighbor_idx = int(self.triangulation.neighbors[tri_idx, j])
                if neighbor_idx != -1:
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

    def expand_from_seed_triangle(self, dual_graph, seed_triangle_idx, max_iterations: int = 100) -> dict:
        """Phase 1 - merge adjacent triangles into P_sub while keeping all three
        seed vertices on the boundary (Ekeland criticality stopping condition).
        """
        original_simplex, _ = self.get_triangle_vertices(seed_triangle_idx)
        original_vertices = set(int(v) for v in original_simplex)

        current_region: Set[int] = {seed_triangle_idx}
        frontier: Set[int] = set(dual_graph[seed_triangle_idx])
        visited: Set[int] = set(dual_graph[seed_triangle_idx]) | {seed_triangle_idx}

        expansion_steps = 0
        for _ in range(max_iterations):
            added_in_this_step = False
            for candidate_tri_idx in sorted(frontier):
                temp_region = current_region | {candidate_tri_idx}
                temp_boundary_edges = self.get_region_boundary_edges(temp_region)
                temp_boundary_vertices = self.get_region_boundary_vertices(temp_region)
                original_still_on_boundary = original_vertices.issubset(temp_boundary_vertices)
                ordered_loop = self.order_boundary_vertices(temp_boundary_edges)
                no_holes = len(ordered_loop) == len(temp_boundary_vertices)

                if original_still_on_boundary and no_holes:
                    current_region = temp_region
                    frontier.remove(candidate_tri_idx)
                    for neighbor_idx in dual_graph[candidate_tri_idx]:
                        if neighbor_idx not in visited:
                            frontier.add(neighbor_idx)
                            visited.add(neighbor_idx)
                    expansion_steps += 1
                    added_in_this_step = True
                    break
            if not added_in_this_step:
                break

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


def build_dual_graph(triangulation) -> Dict[int, List[int]]:
    return TriangulationGraph(triangulation).build_dual_graph()


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
