#!/usr/bin/env python
"""Method-illustration figures: (1) the Ekeland free-cone descriptor geometry,
(2) segmentation + CDT mesh overlay on a real UAV frame. CPU-only.
Outputs outputs/paper/fig_descriptor.{pdf,png} and fig_cdt.{pdf,png}.

Visual design: Google Sans typography, validated CVD-safe categorical palette
(see gstyle.py). Layout tuned so no label overlaps the geometry it annotates.
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

from localization.geometry.descriptor import _interior_simplex_mask
from localization.preprocess.uav import process_uav
import gstyle
gstyle.setup()

FIGDIR = Path("outputs/paper"); FIGDIR.mkdir(parents=True, exist_ok=True)
C_POLY = gstyle.BLUE; C_TRI = "#c3c2b7"; C_SEED = gstyle.ORANGE; C_CONE = gstyle.AQUA


def savefig(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf"); fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    plt.close(fig); print(f"[fig] {name}", flush=True)


def fig_descriptor():
    # An L-shaped footprint with a reflex vertex -> non-trivial Ekeland angle.
    poly = np.array([(0, 0), (6, 0), (6, 2.4), (2.6, 2.4), (2.6, 5.2), (0, 5.2)], float)
    tri = Delaunay(poly)
    mask = _interior_simplex_mask(tri, poly)
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97)
    # CDT mesh (interior triangles)
    for s, keep in zip(tri.simplices, mask):
        if keep:
            ax.add_patch(MplPoly(poly[s], closed=True, fill=False, ec=C_TRI, lw=0.9, zorder=2))
    # seed triangle shaded
    seed = [s for s, k in zip(tri.simplices, mask) if k][0]
    ax.add_patch(MplPoly(poly[seed], closed=True, facecolor=C_SEED, alpha=0.22, ec=C_SEED, lw=1.3, zorder=3))
    # footprint outline
    ax.add_patch(MplPoly(poly, closed=True, fill=False, ec=C_POLY, lw=2.2, zorder=4))
    # Ekeland free cone at the reflex vertex (2.6,2.4): interior 270 -> exterior 90
    vx, vy = 2.6, 2.4
    ax.add_patch(Wedge((vx, vy), 1.5, 0, 90, facecolor=C_CONE, alpha=0.30, ec=C_CONE, lw=1.1, zorder=3))
    ax.annotate(r"$e_v=90^\circ$", (vx + 0.65, vy + 0.75), fontsize=10, color="#0d7a5a", fontweight="medium")
    ax.plot(vx, vy, "o", color=C_CONE, ms=6, zorder=5, mec="white", mew=0.8)
    ax.annotate("reflex vertex", (vx + 0.15, vy - 0.42), fontsize=8, ha="left", color=gstyle.INK_SECONDARY)
    ax.annotate("footprint", (4.35, 0.32), fontsize=8.5, color=C_POLY, fontweight="medium")
    ax.annotate("seed triangle", (0.55, 1.05), fontsize=8, color="#a34e00")
    ax.set_xlim(-0.7, 7.0); ax.set_ylim(-0.9, 6.2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Ekeland free-cone angle $e_v$ at a footprint vertex", fontsize=10.5, pad=10)
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
