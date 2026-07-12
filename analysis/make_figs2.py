#!/usr/bin/env python
"""Method-illustration figures: (1) the 3-panel Ekeland free-cone descriptor
geometry, (2) segmentation + CDT mesh overlay on a real UAV frame. CPU-only.
Outputs outputs/paper/fig_descriptor.{pdf,png} and fig_cdt.{pdf,png}.

Visual design: Google Sans typography, validated CVD-safe categorical palette
(see gstyle.py). Layout tuned so no label overlaps the geometry it annotates.

fig_descriptor is a 3-panel figure (paper Figure 2 / fig:feature_extraction):
(a) convex vertex -- saturates at e_v=180 deg at every depth D_max
(b) shallow expansion -- a nearby recess is revealed after 2 merge steps
(c) deep expansion -- the SAME true recess (same notch, different seed
    triangle) only becomes visible after all 4 merge steps

All angles, P_sub polygons, and seed triangles are computed by the actual
project code (scipy Delaunay + localization.geometry.{triangulation,ekeland})
for the exact coordinates below -- nothing is hand-derived. Verified via
find_examples.py on the notch polygon [(0,0),(3,0),(3,3.2),(1.5,3.2),
(1.5,1.9),(0.9,1.9),(0.9,3.2),(0,3.2)]:
  (a) rectangle, any corner:  e0=180.0  e4=180.0
  (b) seed=5, vertex=5:       e(D=0..4)=[180.0, 180.0, 90.0, 90.0, 90.0]
  (c) seed=1, vertex=4:       e(D=0..4)=[180.0, 180.0, 139.1, 139.1, 90.0]
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Wedge
from scipy.spatial import Delaunay
from shapely.geometry import Polygon as ShapelyPolygon

from localization.geometry.descriptor import _interior_simplex_mask
from localization.geometry.triangulation import (
    build_dual_graph, expand_from_seed_triangle, order_boundary_vertices,
)
from localization.geometry.ekeland import calculate_ekeland_angles_on_boundary
from localization.preprocess.uav import process_uav
import gstyle
gstyle.setup()

FIGDIR = Path("outputs/paper"); FIGDIR.mkdir(parents=True, exist_ok=True)
C_POLY = gstyle.BLUE; C_TRI = "#c3c2b7"; C_SEED = gstyle.ORANGE; C_CONE = gstyle.AQUA
C_PSUB = gstyle.VIOLET


def savefig(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf"); fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    plt.close(fig); print(f"[fig] {name}", flush=True)


def region_boundary_poly(tri, region_triangles):
    """Ordered boundary-vertex polygon (as xy array) for a triangle-index set."""
    edge_count = {}
    for t in region_triangles:
        s = tri.simplices[t]
        for e in [(s[0], s[1]), (s[1], s[2]), (s[2], s[0])]:
            key = tuple(sorted(e))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    ordered = order_boundary_vertices(tri, boundary_edges)
    return tri.points[ordered]


def cone_wedge_angles(tri, region_triangles, vertex_idx):
    """Recompute the exact two boundary-edge directions at ``vertex_idx`` for
    the P_sub formed by ``region_triangles`` -- the same geometry
    calculate_ekeland_angles_on_boundary uses -- so the drawn wedge always
    matches the printed e_v exactly."""
    edge_count = {}
    for t in region_triangles:
        s = tri.simplices[t]
        for e in [(s[0], s[1]), (s[1], s[2]), (s[2], s[0])]:
            key = tuple(sorted(e))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    ordered_boundary = order_boundary_vertices(tri, boundary_edges)
    coords = [tri.points[v] for v in ordered_boundary]
    poly = ShapelyPolygon(coords)
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
    ang_prev = np.degrees(np.arctan2(v_prev[1], v_prev[0])) % 360
    ang_next = np.degrees(np.arctan2(v_next[1], v_next[0])) % 360
    # exterior (free) cone spans from the "next" boundary ray to the "prev" one
    # going the SHORT way around outside the polygon interior
    alpha_int = (ang_prev - ang_next) % 360.0
    alpha_ext = 360.0 - alpha_int
    e_v = min(alpha_ext, 180.0)
    return ang_next, ang_next + e_v, e_v, p_curr


def panel_a(ax):
    poly = np.array([(0, 0), (4, 0), (4, 3), (0, 3)], float)
    tri = Delaunay(poly)
    mask = _interior_simplex_mask(tri, poly)
    dual = build_dual_graph(tri, valid_mask=mask)
    seed_idx = 0
    vtx = int(tri.simplices[seed_idx][0])
    exp4 = expand_from_seed_triangle(tri, dual, seed_idx, max_iterations=4)
    e4 = calculate_ekeland_angles_on_boundary(tri, exp4)
    ev = e4[vtx]["ekeland_angle"]
    assert abs(ev - 180.0) < 1e-6, ev

    for t in range(len(tri.simplices)):
        if mask[t]:
            ax.add_patch(MplPoly(tri.points[tri.simplices[t]], closed=True,
                                  fill=False, ec=C_TRI, lw=0.9, zorder=2))
    ax.add_patch(MplPoly(tri.points[tri.simplices[seed_idx]], closed=True,
                          facecolor=C_SEED, alpha=0.22, ec=C_SEED, lw=1.3, zorder=3))
    ax.add_patch(MplPoly(poly, closed=True, fill=False, ec=C_POLY, lw=2.2, zorder=4))
    a0, a1, _, p = cone_wedge_angles(tri, [seed_idx], vtx)
    ax.add_patch(Wedge(p, 1.35, a0, a1, facecolor=C_CONE, alpha=0.28, ec=C_CONE, lw=1.0, zorder=3))
    ax.plot(*p, "o", color=C_CONE, ms=6, zorder=5, mec="white", mew=0.8)
    ax.annotate(rf"$e_v={ev:.0f}^\circ$", (p[0] - 1.55, p[1] + 0.15), fontsize=10,
                color="#0d7a5a", fontweight="medium", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.annotate("convex vertex", (p[0] - 1.55, p[1] - 0.42), fontsize=8, ha="left",
                color=gstyle.INK_SECONDARY,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(-0.7, 4.7); ax.set_ylim(-0.7, 3.9); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) convex: saturates at $180^\\circ$\nfor every $D_{\\max}$", fontsize=9.6)


def panel_bc(ax, seed_idx, depth, title, poly, vtx):
    poly = np.asarray(poly, float)
    tri = Delaunay(poly)
    mask = _interior_simplex_mask(tri, poly)
    dual = build_dual_graph(tri, valid_mask=mask)
    exp = expand_from_seed_triangle(tri, dual, seed_idx, max_iterations=depth)
    e = calculate_ekeland_angles_on_boundary(tri, exp)
    ev = e[vtx]["ekeland_angle"]
    region = sorted(exp["region_triangles"])

    for t in range(len(tri.simplices)):
        if mask[t]:
            ax.add_patch(MplPoly(tri.points[tri.simplices[t]], closed=True,
                                  fill=False, ec=C_TRI, lw=0.8, zorder=2))
    # P_sub region shaded
    for t in region:
        ax.add_patch(MplPoly(tri.points[tri.simplices[t]], closed=True,
                              facecolor=C_PSUB, alpha=0.16, ec="none", zorder=2.5))
    p_sub_boundary = region_boundary_poly(tri, region)
    ax.add_patch(MplPoly(p_sub_boundary, closed=True, fill=False, ec=C_PSUB, lw=1.6,
                          ls=(0, (4, 2)), zorder=3.4))
    # seed triangle highlighted
    ax.add_patch(MplPoly(tri.points[tri.simplices[seed_idx]], closed=True,
                          facecolor=C_SEED, alpha=0.30, ec=C_SEED, lw=1.3, zorder=3))
    # true footprint outline
    ax.add_patch(MplPoly(poly, closed=True, fill=False, ec=C_POLY, lw=2.0, zorder=4))
    a0, a1, ev2, p = cone_wedge_angles(tri, region, vtx)
    assert abs(ev2 - ev) < 1e-6
    ax.add_patch(Wedge(p, 0.62, a0, a1, facecolor=C_CONE, alpha=0.35, ec=C_CONE, lw=1.0, zorder=5))
    ax.plot(*p, "o", color=C_CONE, ms=6, zorder=6, mec="white", mew=0.8)
    ax.annotate(rf"$e_v={ev:.1f}^\circ$", (p[0] + 0.12, p[1] - 0.55), fontsize=9.5,
                color="#0d7a5a", fontweight="medium", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85, zorder=5.5))
    ax.set_xlim(-0.30, 3.30); ax.set_ylim(-0.30, 3.55); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9.6)
    return region


def fig_descriptor():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.7))
    fig.subplots_adjust(top=0.78, bottom=0.06, left=0.02, right=0.98, wspace=0.14)
    panel_a(axes[0])

    notch = [(0, 0), (3, 0), (3, 3.2), (1.5, 3.2), (1.5, 1.9),
             (0.9, 1.9), (0.9, 3.2), (0, 3.2)]
    r_b = panel_bc(axes[1], seed_idx=5, depth=2,
                   title="(b) shallow: 2 merges already\nreveal a nearby recess",
                   poly=notch, vtx=5)
    r_c = panel_bc(axes[2], seed_idx=1, depth=4,
                   title="(c) deep: the same recess only\nsurfaces at $D_{\\max}{=}4$",
                   poly=notch, vtx=4)
    print(f"[check] panel (b) P_sub triangles={r_b}  panel (c) P_sub triangles={r_c}")

    fig.suptitle("Kernel expansion $P_{\\mathrm{sub}}$: same rule, different convergence depth",
                 fontsize=11.2, y=0.975)
    savefig(fig, "fig_descriptor")


def fig_cdt():
    site = "01"
    sp = Path("outputs/eval") / site
    # a building-dense frame
    name = "01_0001.JPG"
    _, img500, _ = process_uav(f"UAV-VisLoc/{site}/drone/{name}", f"UAV-VisLoc/{site}/{site}.csv")
    polys = json.loads((sp / "uav_polygons" / f"{Path(name).stem}.json").read_text())["polygons"]
    fig, ax = plt.subplots(figsize=(3.8, 4.0))
    fig.subplots_adjust(top=0.91, bottom=0.02, left=0.02, right=0.98)
    ax.imshow(np.asarray(img500))
    for p in polys:
        a = np.asarray(p, float)
        if a.shape[0] < 3:
            continue
        ax.add_patch(MplPoly(a, closed=True, fill=False, ec=C_SEED, lw=1.1, zorder=3))
        try:
            t = Delaunay(a); m = _interior_simplex_mask(t, a)
            for s, k in zip(t.simplices, m):
                if k:
                    ax.add_patch(MplPoly(a[s], closed=True, fill=False, ec=gstyle.BLUE, lw=0.45, alpha=0.75, zorder=2))
        except Exception:
            pass
    ax.set_title(f"Segmentation + CDT mesh ({name})", fontsize=10.5, pad=10)
    ax.axis("off")
    savefig(fig, "fig_cdt")


if __name__ == "__main__":
    fig_descriptor()
    fig_cdt()
    print("[done] fig_descriptor + fig_cdt", flush=True)
