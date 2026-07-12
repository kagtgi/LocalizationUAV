#!/usr/bin/env python
"""Second top-100-vs-GT demo, for site 04 (Taizhou-6, dense urban/industrial) --
same honesty constraints as make_figs5.py (site 01): the frame is the same
best-supported frame picked by an independent frame_support() computation, no
post-hoc cherry-picking for a flattering example. CPU-only.

Outputs outputs/paper/fig_top100_demo_04.{pdf,png}.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from localization.database.kdtree import SatelliteDatabase
from localization.geometry.descriptor import triangle_descriptors_from_polygon
from localization.io.bounds import pixel_offset_to_meters
import gstyle
gstyle.setup()

CENTER = 250.0
SITE = "04"
OUT = Path("outputs/eval") / SITE
FIGDIR = Path("outputs/paper"); FIGDIR.mkdir(parents=True, exist_ok=True)
K = 100
COV_RADIUS_PX = 500.0


def load_sequence(limit=100):
    rec = pd.read_csv(OUT / "records_main.csv", dtype={"image": str})
    rec = rec[rec["gt_px_x"].notna() & rec["gt_px_y"].notna()].sort_values("image").head(limit)
    frames = []
    for _, r in rec.iterrows():
        jf = OUT / "uav_polygons" / f"{Path(str(r['image'])).stem}.json"
        if jf.exists():
            frames.append((str(r["image"]), json.loads(jf.read_text())["polygons"],
                           float(r["gt_px_x"]), float(r["gt_px_y"])))
    return frames


def frame_support(db, polys, r_obs, k=5):
    descs, cents = [], []
    for p in polys:
        d, c = triangle_descriptors_from_polygon(p, max_depth=4)
        if d.shape[0]:
            descs.append(d); cents.append(c)
    if not descs:
        return np.zeros((0, 2))
    desc = np.vstack(descs).astype(np.float32); cent = np.vstack(cents)
    _, idx = db.query(desc, k=min(k, db.size), p=1)
    if idx.ndim == 1:
        idx = idx[:, None]
    sat = db.centroids[idx.reshape(-1)]
    off = np.repeat(cent - CENTER, idx.shape[1], axis=0)
    return sat - off


def top100_candidates(db, polys):
    descs, cents = [], []
    for p in polys:
        d, c = triangle_descriptors_from_polygon(p, max_depth=4)
        if d.shape[0]:
            descs.append(d); cents.append(c)
    if not descs:
        return np.zeros((0, 2)), 0
    desc = np.vstack(descs).astype(np.float32); cent = np.vstack(cents)
    n_tri = desc.shape[0]
    _, idx = db.query(desc, k=min(K, db.size), p=1)
    if idx.ndim == 1:
        idx = idx[:, None]
    sat = db.centroids[idx.reshape(-1)]
    off = np.repeat(cent - CENTER, idx.shape[1], axis=0)
    return sat - off, n_tri


def main():
    meta = json.loads((OUT / "site_meta.json").read_text())
    bounds, sat_w, sat_h = meta["bounds"], meta["sat_width"], meta["sat_height"]
    db = SatelliteDatabase.load(str(OUT / "satellite_kdtree.npz"))
    frames = load_sequence(100)

    support = []
    for _, polys, gx_i, gy_i in frames:
        v = frame_support(db, polys, 150.0)
        support.append(len(cKDTree(v).query_ball_point([gx_i, gy_i], 150.0)) if v.shape[0] >= 3 else 0)
    bi = int(np.argmax(support))
    name, polys, gx, gy = frames[bi]

    cand, n_tri = top100_candidates(db, polys)
    d = np.hypot(cand[:, 0] - gx, cand[:, 1] - gy)
    n_near = int(np.sum(d < COV_RADIUS_PX))
    med_err_m = pixel_offset_to_meters(float(np.median(cand[:, 0])) - gx,
                                        float(np.median(cand[:, 1])) - gy,
                                        bounds, sat_w, sat_h)["distance_m"]
    print(f"[demo04] frame={name} n_triangles={n_tri} n_candidates={cand.shape[0]} "
          f"within {COV_RADIUS_PX:.0f}px of GT: {n_near}/{cand.shape[0]}", flush=True)

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(Path("UAV-VisLoc") / SITE / f"satellite{SITE}.tif") as im:
        scale = 1600.0 / max(im.size)
        prev = im.resize((int(im.size[0] * scale), int(im.size[1] * scale))).convert("RGB")

    fig = plt.figure(figsize=(8.2, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.28,
                          left=0.03, right=0.98, top=0.86, bottom=0.14)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(prev)
    ax0.scatter(cand[:, 0] * scale, cand[:, 1] * scale, s=10, c=gstyle.ORANGE,
               alpha=0.55, lw=0, zorder=3, label=f"top-{K} candidates/triangle ({cand.shape[0]} total)")
    ax0.scatter([gx * scale], [gy * scale], s=140, marker="*", c=gstyle.GOOD,
               edgecolor="white", lw=0.8, zorder=5, label="ground truth")
    ax0.axis("off")
    ax0.set_title("(a) all candidate implied positions", fontsize=9.3)
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7.4, frameon=True,
              framealpha=0.92, markerscale=1.3, edgecolor="none", facecolor="white")

    ax1 = fig.add_subplot(gs[0, 1])
    uav_img = Image.open(Path("UAV-VisLoc") / SITE / "drone" / name)
    ax1.imshow(np.asarray(uav_img))
    ax1.axis("off")
    ax1.set_title("(b) UAV query image", fontsize=9.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(prev)
    ax2.scatter(cand[:, 0] * scale, cand[:, 1] * scale, s=18, c=gstyle.ORANGE,
               alpha=0.7, lw=0, zorder=3)
    ax2.scatter([gx * scale], [gy * scale], s=200, marker="*", c=gstyle.GOOD,
               edgecolor="white", lw=1.0, zorder=5)
    zoom_px = 1800
    ax2.set_xlim((gx - zoom_px) * scale, (gx + zoom_px) * scale)
    ax2.set_ylim((gy + zoom_px) * scale, (gy - zoom_px) * scale)
    ax2.axis("off")
    pct_near = 100.0 * n_near / cand.shape[0]
    ax2.set_title(f"(c) zoom at GT: {n_near}/{cand.shape[0]} pooled\ncandidates within {COV_RADIUS_PX:.0f} px ({pct_near:.1f}%)",
                 fontsize=9.3)

    fig.suptitle(f"Site 04 (Taizhou-6, urban/industrial), frame {name}: the {K}-nearest-descriptor\n"
                 f"candidates from every query triangle scatter across the map, not toward the true position",
                 fontsize=10, y=0.99)
    fig.savefig(FIGDIR / "fig_top100_demo_04.pdf")
    fig.savefig(FIGDIR / "fig_top100_demo_04.png", dpi=300)
    print(f"[fig] fig_top100_demo_04 written (median candidate error {med_err_m:.0f}m)", flush=True)

    (FIGDIR / "top100_demo_04_metrics.json").write_text(json.dumps({
        "site": SITE, "frame": name, "n_triangles": n_tri, "n_candidates": int(cand.shape[0]),
        "K": K, "coverage_radius_px": COV_RADIUS_PX, "n_within_radius": n_near,
        "median_candidate_error_m": med_err_m,
    }, indent=2))


if __name__ == "__main__":
    main()
