"""Standalone visualization of one CDT triangle expansion + Ekeland angle.

Useful for sanity-checking the geometry pipeline on a synthetic polygon.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MatplotlibPolygon
from matplotlib.patches import Wedge
from scipy.spatial import Delaunay

from .ekeland import calculate_ekeland_angles_on_boundary
from .triangulation import build_dual_graph, expand_from_seed_triangle


def create_sample_triangulation():
    outer = np.array([[0, 0], [4, -1], [8, 0], [10, 4], [8, 8], [3, 9], [0, 6], [-2, 3]])
    inner = np.array([[2, 3], [5, 2], [6, 5], [3, 6], [4, 4]])
    return Delaunay(np.vstack((outer, inner)))


def plot_expansion(seed_idx: int = 0, output_dir: str = "outputs/expansion"):
    os.makedirs(output_dir, exist_ok=True)
    tri = create_sample_triangulation()
    dual = build_dual_graph(tri)
    expansion = expand_from_seed_triangle(tri, dual, seed_idx, max_iterations=500)
    ekeland = calculate_ekeland_angles_on_boundary(tri, expansion)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.triplot(
        tri.points[:, 0],
        tri.points[:, 1],
        tri.simplices,
        color="gray",
        linestyle=":",
        linewidth=0.8,
        alpha=0.6,
    )

    seed_pts = tri.points[tri.simplices[seed_idx]]
    ax.add_patch(
        MatplotlibPolygon(
            seed_pts,
            facecolor="cyan",
            edgecolor="dodgerblue",
            alpha=0.4,
            linewidth=2,
            label="Target Triangle",
        )
    )

    for idx, (v1, v2) in enumerate(expansion["boundary_edges"]):
        p1, p2 = tri.points[v1], tri.points[v2]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="black",
            linewidth=1.0,
            label="Expanded Region" if idx == 0 else None,
        )

    for v_idx, data in ekeland.items():
        angle = data["ekeland_angle"]
        if angle <= 0:
            continue
        pos = tri.points[v_idx]
        best_axis = data["best_axis"]
        bisector_angle = np.degrees(np.arctan2(best_axis[1], best_axis[0]))
        ax.add_patch(
            Wedge(
                pos,
                r=0.6,
                theta1=bisector_angle - angle / 2,
                theta2=bisector_angle + angle / 2,
                color="red",
                alpha=0.65,
                zorder=5,
            )
        )
        text_x = pos[0] + 0.8 * np.cos(np.radians(bisector_angle))
        text_y = pos[1] + 0.8 * np.sin(np.radians(bisector_angle))
        ax.text(text_x, text_y, f"{int(angle)} deg", fontweight="bold", fontsize=10, ha="center", va="center", zorder=6)

    ax.set_aspect("equal")
    ax.set_title("Ekeland Angle and Region Expansion", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    save_path = os.path.join(output_dir, f"expansion_seed_{seed_idx}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return save_path


if __name__ == "__main__":
    plot_expansion()
