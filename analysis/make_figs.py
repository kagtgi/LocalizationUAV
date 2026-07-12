#!/usr/bin/env python
"""Consolidate OURS metrics + generate journal-ready figures (CPU, headless).
Runs on the VM in ~/LocalizationUAV against cached site-01 artifacts.
Outputs: outputs/paper/metrics.json and outputs/paper/fig_{disc,scale,seq,votecloud}.{pdf,png}

Visual design: Google Sans typography, validated CVD-safe categorical palette,
thin marks, hairline recessive grid/axes, generous padding (see gstyle.py).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from localization.database.kdtree import SatelliteDatabase
from localization.geometry.descriptor import triangle_descriptors_from_polygon
from localization.io.bounds import pixel_offset_to_meters
import gstyle
gstyle.setup()

CENTER = 250.0
SITE = "01"
OUT = Path("outputs/eval") / SITE
FIGDIR = Path("outputs/paper"); FIGDIR.mkdir(parents=True, exist_ok=True)

C_ANGLE = gstyle.BLUE
C_RET = gstyle.ORANGE
C_ERR = gstyle.RED
C_OK = gstyle.GOOD
C_GREY = gstyle.INK_MUTED


def savefig(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf"); fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"[fig] wrote {name}.pdf/.png", flush=True)


def frame_votes(db, polys, k=5):
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


def main():
    meta = json.loads((OUT / "site_meta.json").read_text())
    bounds, sat_w, sat_h = meta["bounds"], meta["sat_width"], meta["sat_height"]
    db = SatelliteDatabase.load(str(OUT / "satellite_kdtree.npz"))
    frames = load_sequence(100)
    print(f"[load] db={db.size} triangles, {len(frames)} frames", flush=True)

    metrics = {"site": SITE, "n_triangles": int(db.size), "n_frames": len(frames)}

    # ---- Fig A: discriminability ceiling (measured, stable numbers) ----
    variants = ["5-D\nangles", "+canon.", "+z-score", "+size", "+context", "bldg.", "bldg.\n+context"]
    truedist = [38.0, 18.1, 2.4, 4.3, np.nan, np.nan, np.nan]
    retrieval = [0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.3]
    metrics["discriminability"] = {"variants": [v.replace("\n", " ") for v in variants],
                                   "true_match_L1": truedist, "retrieval_pct": retrieval,
                                   "coverage_pct": 92.9}
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    fig.subplots_adjust(wspace=0.38, bottom=0.30, top=0.86)
    xs = np.arange(4)
    a0.bar(xs, truedist[:4], color=C_ANGLE, width=0.58, zorder=3)
    gstyle.clean_axes(a0, grid_axis="y")
    a0.set_xticks(xs); a0.set_xticklabels([v.replace("\n", " ") for v in variants[:4]], fontsize=8.3)
    a0.set_ylabel(r"true-match $\ell_1$ distance")
    a0.set_ylim(0, max(truedist[:4]) * 1.22)
    a0.set_title("(a) enrichment shrinks the true-match\ndistance ($16\\times$)", fontsize=9.5)
    for i, v in enumerate(truedist[:4]):
        gstyle.annotate(a0, i, v, f"{v:.1f}", color=gstyle.INK_SECONDARY, fontsize=8, weight="medium")

    xr = np.arange(len(variants))
    a1.bar(xr, retrieval, color=C_RET, width=0.58, zorder=3)
    gstyle.clean_axes(a1, grid_axis="y")
    a1.set_xticks(xr); a1.set_xticklabels([v.replace("\n", " ") for v in variants],
                                           fontsize=7.6, rotation=32, ha="right")
    a1.set_ylabel("retrieval @ top-$K$ (%)"); a1.set_ylim(0, 5.4)
    a1.axhline(1.4, ls="-", lw=0.9, color=C_GREY, zorder=2)
    a1.annotate("chance", (len(variants) - 1, 1.75), ha="right", va="bottom",
                fontsize=7.8, color=C_GREY)
    a1.set_title("(b) yet retrieval stays $\\approx$0%", fontsize=9.5)
    savefig(fig, "fig_disc")

    # ---- Fig B: scale dependence (ROI sweep, from db descriptors, z-scored) ----
    mu, sd = db.descriptors.mean(0), db.descriptors.std(0); sd[sd < 1e-6] = 1.0
    satn = (db.descriptors - mu) / sd
    cen_tree = cKDTree(db.centroids)
    radii_px = [833, 1667, 3333, 6667, 13334]  # 250m..4km @0.3m/px
    roi_err = []
    for R in radii_px:
        errs = []
        for name, polys, gx, gy in frames[:60]:
            descs, cents = [], []
            for p in polys:
                d, c = triangle_descriptors_from_polygon(p, max_depth=4)
                if d.shape[0]:
                    descs.append(d); cents.append(c)
            if not descs:
                continue
            d = (np.vstack(descs).astype(np.float32) - mu) / sd
            c = np.vstack(cents)
            roi = np.asarray(cen_tree.query_ball_point([gx, gy], R), dtype=int)
            if roi.size < 6:
                continue
            lt = cKDTree(satn[roi])
            _, loc = lt.query(d, k=min(5, roi.size))
            if loc.ndim == 1:
                loc = loc[:, None]
            gi = roi[loc]
            votes = (db.centroids[gi.reshape(-1)] - np.repeat(c - CENTER, loc.shape[1], axis=0))
            keys = np.floor(votes / 200.0).astype(np.int64)
            _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            pred = votes[inv == int(cnt.argmax())].mean(0)
            errs.append(pixel_offset_to_meters(pred[0] - gx, pred[1] - gy, bounds, sat_w, sat_h)["distance_m"])
        roi_err.append(float(np.median(errs)) if errs else np.nan)
    km2 = [np.pi * (r * 0.3 / 1000.0) ** 2 for r in radii_px]  # circle area km^2
    metrics["scale_sweep"] = {"radius_px": radii_px, "search_area_km2": km2, "median_err_m": roi_err,
                              "full_map_err_m": 3492.0}
    fig, ax = plt.subplots(figsize=(3.9, 3.1))
    fig.subplots_adjust(bottom=0.18, top=0.82, left=0.20, right=0.95)
    gstyle.clean_axes(ax, grid_axis="y")
    ax.plot(km2, roi_err, "o-", color=C_ERR, lw=1.6, ms=5.5, zorder=3, label="constrained search")
    ax.axhline(3492, ls="-", lw=1.0, color=C_GREY, zorder=2, label="full map ($\\approx$100 km$^2$)")
    ax.set_xscale("log"); ax.set_xlabel("search area (km$^2$)"); ax.set_ylabel("median error (m)")
    ax.set_title("Error is set by the search window,\nnot by identification", fontsize=9.5)
    ax.legend(loc="upper left", handlelength=1.6)
    savefig(fig, "fig_scale")

    # ---- Fig C: sequential support + particle-filter convergence ----
    rng = np.random.default_rng(0)
    N = 20000
    px = rng.uniform(0, sat_w, N); py = rng.uniform(0, sat_h, N); w = np.full(N, 1.0 / N)
    support, sf_err, pf_err = [], [], []
    r_obs = 150.0
    for name, polys, gx, gy in frames:
        v = frame_votes(db, polys, 5)
        # per-frame single-frame Hough error + GT support
        if v.shape[0] >= 3:
            vt = cKDTree(v)
            support.append(len(vt.query_ball_point([gx, gy], r_obs)))
            keys = np.floor(v / 200.0).astype(np.int64)
            _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            sp = v[inv == int(cnt.argmax())].mean(0)
            sf_err.append(pixel_offset_to_meters(sp[0] - gx, sp[1] - gy, bounds, sat_w, sat_h)["distance_m"])
        else:
            support.append(0); sf_err.append(np.nan)
        # particle filter step
        px += rng.normal(0, 266, N); py += rng.normal(0, 266, N)
        ri = rng.choice(N, int(0.05 * N), replace=False)
        px[ri] = rng.uniform(0, sat_w, len(ri)); py[ri] = rng.uniform(0, sat_h, len(ri))
        px = np.clip(px, 0, sat_w); py = np.clip(py, 0, sat_h)
        if v.shape[0] >= 3:
            vt = cKDTree(v)
            cnt = np.array(vt.query_ball_point(np.column_stack([px, py]), r_obs, return_length=True), float)
            w = w * (0.1 + cnt)
        s = w.sum(); w = w / s if s > 0 else np.full(N, 1.0 / N)
        ex, ey = float(np.sum(w * px)), float(np.sum(w * py))
        pf_err.append(pixel_offset_to_meters(ex - gx, ey - gy, bounds, sat_w, sat_h)["distance_m"])
        ess = 1.0 / np.sum(w ** 2)
        if ess < N / 2:
            c = np.cumsum(w); c[-1] = 1.0
            idx = np.searchsorted(c, (rng.random() + np.arange(N)) / N)
            px, py = px[idx] + rng.normal(0, 133, N), py[idx] + rng.normal(0, 133, N)
            w = np.full(N, 1.0 / N)
    support = np.array(support)
    metrics["sequential"] = {"pct_frames_gt_supported": float(100 * (support > 0).mean()),
                             "median_support": float(np.median(support)),
                             "single_frame_median_err_m": float(np.nanmedian(sf_err)),
                             "pf_final_median_err_m": float(np.median(pf_err[-30:]))}
    fig, (b0, b1) = plt.subplots(2, 1, figsize=(4.0, 4.0), sharex=True)
    fig.subplots_adjust(hspace=0.45, bottom=0.11, top=0.88, left=0.17, right=0.96)
    fr = np.arange(len(support))
    gstyle.clean_axes(b0, grid_axis="y")
    b0.bar(fr, support, color=C_ANGLE, width=1.0, zorder=3)
    b0.set_ylabel("votes at\ntrue pos.")
    b0.set_title(f"(a) true position supported in only "
                 f"{100*(support>0).mean():.0f}% of frames", fontsize=9.3)
    gstyle.clean_axes(b1, grid_axis="y")
    b1.plot(fr, np.array(pf_err) / 1000.0, color=C_ERR, lw=1.4, zorder=3, label="particle filter")
    b1.plot(fr, np.array(sf_err) / 1000.0, color=C_GREY, lw=0.9, alpha=0.75, zorder=2, label="single frame")
    b1.set_ylabel("error (km)"); b1.set_xlabel("frame along trajectory")
    b1.set_title("(b) the filter never converges", fontsize=9.3)
    b1.legend(loc="upper right", handlelength=1.6, frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="none")
    savefig(fig, "fig_seq")

    # ---- Fig D: vote cloud on the satellite map for one frame ----
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    # pick the frame with the most GT support (best case) for an honest but legible example
    bi = int(np.argmax(support))
    name, polys, gx, gy = frames[bi]
    v = frame_votes(db, polys, 5)
    with Image.open(Path("UAV-VisLoc") / SITE / f"satellite{SITE}.tif") as im:
        scale = 1500.0 / max(im.size)
        prev = im.resize((int(im.size[0] * scale), int(im.size[1] * scale))).convert("RGB")
    fig, ax = plt.subplots(figsize=(3.6, 3.6 * sat_h / sat_w + 0.4))
    fig.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98)
    ax.imshow(prev)
    ax.scatter(v[:, 0] * scale, v[:, 1] * scale, s=3, c=C_RET, alpha=0.4, lw=0, label="descriptor votes")
    ax.scatter([gx * scale], [gy * scale], s=100, marker="*", c=C_OK, edgecolor="white",
               lw=0.7, label="true position", zorder=5)
    ax.set_title(f"Frame {name} (best-supported)", fontsize=9.3)
    ax.axis("off")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, markerscale=1.6, fontsize=8.2,
              edgecolor="none", facecolor="white")
    savefig(fig, "fig_votecloud")

    (FIGDIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[done] metrics.json + 4 figures in outputs/paper/", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
