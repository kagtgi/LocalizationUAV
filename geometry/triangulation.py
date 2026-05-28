from collections import deque
import numpy as np

class TriangulationGraph:
    """Dual-graph and region expansion operations for Delaunay triangulation."""

    def __init__(self, triangulation):
        self.triangulation = triangulation

    def build_dual_graph(self):
        dual_graph = {i: [] for i in range(len(self.triangulation.simplices))}

        for tri_idx in range(len(self.triangulation.simplices)):
            for j in range(3):
                neighbor_idx = self.triangulation.neighbors[tri_idx, j]
                if neighbor_idx != -1:
                    dual_graph[tri_idx].append(neighbor_idx)

        return dual_graph

    def get_triangle_vertices(self, triangle_idx):
        simplex = self.triangulation.simplices[triangle_idx]
        vertices = self.triangulation.points[simplex]
        return simplex, vertices

    def get_region_boundary_edges(self, region_triangle_indices):
        edge_count = {}

        for tri_idx in region_triangle_indices:
            simplex = self.triangulation.simplices[tri_idx]
            edges = [
                tuple(sorted([simplex[0], simplex[1]])),
                tuple(sorted([simplex[1], simplex[2]])),
                tuple(sorted([simplex[2], simplex[0]])),
            ]

            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1

        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return boundary_edges

    def get_region_boundary_vertices(self, region_triangle_indices):
        boundary_edges = self.get_region_boundary_edges(region_triangle_indices)
        boundary_vertices = set()

        for v1, v2 in boundary_edges:
            boundary_vertices.add(v1)
            boundary_vertices.add(v2)

        return boundary_vertices

    def order_boundary_vertices(self, boundary_edges):
        if not boundary_edges:
            return []

        adjacency = {}
        for v1, v2 in boundary_edges:
            adjacency.setdefault(v1, []).append(v2)
            adjacency.setdefault(v2, []).append(v1)

        start_vertex = boundary_edges[0][0]
        ordered = [start_vertex]
        visited = {start_vertex}

        current = start_vertex
        while len(ordered) < len(adjacency):
            neighbors = adjacency[current]
            next_vertex = None

            for neighbor in neighbors:
                if neighbor not in visited:
                    next_vertex = neighbor
                    break

            if next_vertex is None:
                break

            ordered.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex

        return ordered
    
    def is_convex_polygon(self, ordered_vertex_indices):
        if len(ordered_vertex_indices) <= 3:
            return True 
            
        coords = [self.triangulation.points[idx] for idx in ordered_vertex_indices]
        n = len(coords)
        
        sign = 0
        for i in range(n):
            p_prev = coords[i - 1]
            p_curr = coords[i]
            p_next = coords[(i + 1) % n]
            
            dx1, dy1 = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
            dx2, dy2 = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
            
            cross_product = dx1 * dy2 - dy1 * dx2
            
            curr_sign = np.sign(cross_product)
            
            if curr_sign == 0:
                continue
                
            if sign == 0:
                sign = curr_sign
            elif sign != curr_sign:
                return False 
                
        return True
    
    def check_convexity_and_get_concave_verts(self, ordered_loop):
        if len(ordered_loop) <= 3:
            return True, []
            
        coords = [self.triangulation.points[idx] for idx in ordered_loop]
        n = len(coords)
        
        signed_area = 0
        for i in range(n):
            p1, p2 = coords[i], coords[(i + 1) % n]
            signed_area += (p1[0] * p2[1] - p2[0] * p1[1])
        is_ccw = signed_area > 0  
        
        concave_verts = []
        for i in range(n):
            p_prev = coords[i - 1]
            p_curr = coords[i]
            p_next = coords[(i + 1) % n]
            
            dx1, dy1 = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
            dx2, dy2 = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
            
            cross_product = dx1 * dy2 - dy1 * dx2
            
            if is_ccw and cross_product < -1e-6:
                concave_verts.append(ordered_loop[i])
            elif not is_ccw and cross_product > 1e-6:
                concave_verts.append(ordered_loop[i])
                
        is_convex = len(concave_verts) == 0
        return is_convex, concave_verts

    def expand_from_seed_triangle(self, dual_graph, seed_triangle_idx, max_iterations=500):
        original_simplex, _ = self.get_triangle_vertices(seed_triangle_idx)
        original_vertices = set(original_simplex)

        current_region = {seed_triangle_idx}
        
        # Tiền tuyến (Frontier) chứa các tam giác láng giềng sẵn sàng để thử gộp
        frontier = set(dual_graph[seed_triangle_idx])
        visited = set(dual_graph[seed_triangle_idx]) | {seed_triangle_idx}

        expansion_steps = 0

        for _ in range(max_iterations):
            added_in_this_step = False
            
            for candidate_tri_idx in sorted(list(frontier)):
                temp_region = current_region | {candidate_tri_idx}
                
                temp_boundary_edges = self.get_region_boundary_edges(temp_region)
                temp_boundary_vertices = self.get_region_boundary_vertices(temp_region)

                original_still_on_boundary = original_vertices.issubset(temp_boundary_vertices)

                ordered_loop = self.order_boundary_vertices(temp_boundary_edges)
                no_holes = (len(ordered_loop) == len(temp_boundary_vertices))

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
            'region_triangles': current_region,
            'boundary_vertices': current_boundary_vertices,
            'boundary_edges': final_boundary_edges,
            'original_vertices': original_vertices,
            'expansion_steps': expansion_steps,
            'seed_triangle': seed_triangle_idx,
        }
    

def build_dual_graph(triangulation):
    return TriangulationGraph(triangulation).build_dual_graph()


def get_triangle_vertices(triangulation, triangle_idx):
    return TriangulationGraph(triangulation).get_triangle_vertices(triangle_idx)


def get_region_boundary_edges(triangulation, region_triangle_indices):
    return TriangulationGraph(triangulation).get_region_boundary_edges(region_triangle_indices)


def get_region_boundary_vertices(triangulation, region_triangle_indices):
    return TriangulationGraph(triangulation).get_region_boundary_vertices(region_triangle_indices)


def order_boundary_vertices(triangulation, boundary_edges):
    return TriangulationGraph(triangulation).order_boundary_vertices(boundary_edges)


def expand_from_seed_triangle(triangulation, dual_graph, seed_triangle_idx, max_iterations=100):
    return TriangulationGraph(triangulation).expand_from_seed_triangle(
        dual_graph=dual_graph,
        seed_triangle_idx=seed_triangle_idx,
        max_iterations=max_iterations,
    )
