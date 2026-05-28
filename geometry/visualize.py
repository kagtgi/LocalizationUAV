import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotlibPolygon, Wedge
from scipy.spatial import Delaunay

from geometry.triangulation import build_dual_graph, expand_from_seed_triangle
from geometry.ekeland import calculate_ekeland_angles_on_boundary

def create_sample_triangulation():
    """Create a sample triangulation with a mix of outer and inner points."""
    outer_points = np.array([
        [0, 0], [4, -1], [8, 0], [10, 4], 
        [8, 8], [3, 9], [0, 6], [-2, 3]
    ])
    inner_points = np.array([
        [2, 3], [5, 2], [6, 5], [3, 6], [4, 4]
    ])
    
    all_points = np.vstack((outer_points, inner_points))
    triangulation = Delaunay(all_points)
    return triangulation

def plot_and_save_expansion():
    output_dir = "output_visuals"
    os.makedirs(output_dir, exist_ok=True)

    tri = create_sample_triangulation()
    dual_graph = build_dual_graph(tri)
    
    seed_idx = 0
    
    expansion_result = expand_from_seed_triangle(tri, dual_graph, seed_idx, max_iterations=500)

    ekeland_results = calculate_ekeland_angles_on_boundary(tri, expansion_result)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices, 
               color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    seed_pts = tri.points[tri.simplices[seed_idx]]
    seed_poly = MatplotlibPolygon(seed_pts, facecolor='cyan', edgecolor='dodgerblue', 
                                  alpha=0.4, linewidth=2, label='Target Triangle')
    ax.add_patch(seed_poly)

    boundary_edges = expansion_result['boundary_edges']
    for idx, (v1, v2) in enumerate(boundary_edges):
        p1, p2 = tri.points[v1], tri.points[v2]
        if idx == 0:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1.0, label='Expanded Region')
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1.0)

    for v_idx, data in ekeland_results.items():
        ek_angle = data['ekeland_angle']
        if ek_angle > 0:
            pos = tri.points[v_idx]
            
            best_axis = data['best_axis']
            
            bisector_angle = np.degrees(np.arctan2(best_axis[1], best_axis[0]))
            
            wedge = Wedge(pos, r=0.6, 
                          theta1=bisector_angle - ek_angle/2, 
                          theta2=bisector_angle + ek_angle/2, 
                          color='red', alpha=0.65, zorder=5)
            ax.add_patch(wedge)
            
            arrow_length = 1.2 
            dx = best_axis[0] * arrow_length
            dy = best_axis[1] * arrow_length
            ax.arrow(pos[0], pos[1], dx, dy, 
                     head_width=0.15, head_length=0.2, 
                     fc='blue', ec='blue', linestyle='--', linewidth=0.25,
                     zorder=4, length_includes_head=True)
            
            text_x = pos[0] + 0.8 * np.cos(np.radians(bisector_angle))
            text_y = pos[1] + 0.8 * np.sin(np.radians(bisector_angle))
            ax.text(text_x, text_y, f"{int(ek_angle)}°", 
                    fontweight='bold', fontsize=10, ha='center', va='center', zorder=6)
            
    ax.set_aspect('equal')
    ax.set_title("Ekeland Angle and Region Expansion", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    ax.set_xticks([])
    ax.set_yticks([])

    save_path = os.path.join(output_dir, f"expansion_seed_{seed_idx}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Image saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_and_save_expansion()

# PYTHONPATH=. python ./geometry/visualize.py