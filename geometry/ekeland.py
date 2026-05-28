import numpy as np
from shapely.geometry import Polygon

from .triangulation import TriangulationGraph, build_dual_graph, expand_from_seed_triangle, order_boundary_vertices

def get_exact_outward_bisector(triangulation, ordered_boundary, vertex_idx):
    """Tính toán vector phân giác hướng ra ngoài chính xác tuyệt đối cho một đỉnh."""
    coords = [triangulation.points[v] for v in ordered_boundary]
    poly = Polygon(coords)
    
    if not poly.exterior.is_ccw:
        ordered_boundary = ordered_boundary[::-1]
        coords = coords[::-1]

    idx = ordered_boundary.index(vertex_idx)
    n = len(ordered_boundary)
    
    p_prev = np.array(coords[(idx - 1) % n])
    p_curr = np.array(coords[idx])
    p_next = np.array(coords[(idx + 1) % n])
    
    e1 = p_curr - p_prev
    e2 = p_next - p_curr
    
    n1 = np.array([e1[1], -e1[0]])
    n2 = np.array([e2[1], -e2[0]])
    
    n1 = n1 / (np.linalg.norm(n1) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2) + 1e-10)
    
    outward_normal = n1 + n2
    norm_length = np.linalg.norm(outward_normal)
    
    if norm_length < 1e-6:
        outward_normal = n1
    else:
        outward_normal = outward_normal / norm_length
        
    return outward_normal


class EkelandAnalyzer:
    """Class-based Ekeland angle analyzer using O(1) analytical formula."""

    def __init__(self):
        pass

    def calculate_ekeland_angles_on_boundary(self, triangulation, expansion_result):
        """Tính góc Ekeland dựa vào định lý nội tại của đa giác (O(1))."""
        boundary_edges = expansion_result['boundary_edges']
        original_vertices = expansion_result['original_vertices']
        ordered_boundary = order_boundary_vertices(triangulation, boundary_edges)

        if len(ordered_boundary) < 3:
            return {
                v: {'ekeland_angle': 0.0, 'best_axis': np.array([1, 0]), 'axis_name': 'East'}
                for v in original_vertices
            }

        ekeland_results = {}

        for vertex_idx in original_vertices:
            # Nếu đỉnh gốc bị nuốt chửng (không nằm trên biên) -> Góc = 0
            if vertex_idx not in ordered_boundary:
                ekeland_results[vertex_idx] = {
                    'ekeland_angle': 0.0,
                    'best_axis': np.array([1, 0]),
                    'axis_name': 'Interior'
                }
                continue

            exact_axis = get_exact_outward_bisector(triangulation, ordered_boundary, vertex_idx)
            
            coords = [triangulation.points[v] for v in ordered_boundary]
            poly = Polygon(coords)
            
            temp_boundary = ordered_boundary[:]
            if not poly.exterior.is_ccw:
                temp_boundary = temp_boundary[::-1]
                coords = coords[::-1]

            idx = temp_boundary.index(vertex_idx)
            n = len(temp_boundary)
            
            p_prev = np.array(coords[(idx - 1) % n])
            p_curr = np.array(coords[idx])
            p_next = np.array(coords[(idx + 1) % n])
            
            v_prev = p_prev - p_curr
            v_next = p_next - p_curr
            
            angle_prev = np.degrees(np.arctan2(v_prev[1], v_prev[0]))
            angle_next = np.degrees(np.arctan2(v_next[1], v_next[0]))
            
            alpha_int = (angle_prev - angle_next) % 360.0
            
            alpha_ext = 360.0 - alpha_int
            
            ekeland_angle = min(alpha_ext, 180.0)

            ekeland_results[vertex_idx] = {
                'ekeland_angle': ekeland_angle,
                'best_axis': exact_axis,
                'axis_name': 'Analytical Formula',
            }
            
        return ekeland_results

    def compute_expansion_ekeland_for_all_triangles(self, triangulation):
        """Khởi chạy quét Ekeland cho toàn bộ bản đồ."""
        dual_graph = build_dual_graph(triangulation)
        results = []

        num_triangles = len(triangulation.simplices)
        print(f'   Processing {num_triangles} seed triangles (Ekeland O(1))...')

        for seed_idx in range(num_triangles):
            expansion_result = expand_from_seed_triangle(
                triangulation,
                dual_graph,
                seed_idx,
                max_iterations=100, 
            )

            ekeland_results = self.calculate_ekeland_angles_on_boundary(
                triangulation,
                expansion_result
            )

            original_simplex = triangulation.simplices[seed_idx]
            original_coords = triangulation.points[original_simplex]
            ekeland_angles_dict = {v: ekeland_results[v]['ekeland_angle'] for v in ekeland_results}

            results.append(
                {
                    'seed_triangle_idx': seed_idx,
                    'seed_vertices': original_simplex.tolist(),
                    'seed_coordinates': original_coords.tolist(),
                    'ekeland_angles': ekeland_angles_dict,
                    'ekeland_full_results': ekeland_results,
                    'expansion_steps': expansion_result['expansion_steps'],
                    'num_triangles_in_region': len(expansion_result['region_triangles']),
                    'num_boundary_vertices': len(expansion_result['boundary_vertices']),
                }
            )

        print(f'   Completed Ekeland expansion analysis for all {num_triangles} triangles')
        return results

_DEFAULT_ANALYZER = EkelandAnalyzer()

def calculate_ekeland_angles_on_boundary(triangulation, expansion_result, **kwargs):
    return _DEFAULT_ANALYZER.calculate_ekeland_angles_on_boundary(
        triangulation=triangulation,
        expansion_result=expansion_result
    )

def compute_expansion_ekeland_for_all_triangles(triangulation, **kwargs):
    return _DEFAULT_ANALYZER.compute_expansion_ekeland_for_all_triangles(
        triangulation=triangulation
    )