#!/usr/bin/env python
"""Phase A - building-constellation CONTEXT descriptor test (CPU-only, cached data).

Attacks the discriminability ceiling directly: the 5-D per-triangle descriptor is
too ambiguous at city scale (retrieval 0% even after z-score/size). A single
triangle is inherently non-unique; what makes a LOCATION unique is the spatial
arrangement of buildings around it. So we append, to each triangle, a log-polar
histogram of neighbouring building centroids (a Belongie shape-context computed
over buildings rather than boundary points). Yaw alignment + scale calibration
make bearings and distances directly comparable UAV<->satellite.

Reports the same A1 retrievability / A3 Hough-error metrics as phase_a_diag, at
FULL map scale, with vs without context so the effect is unambiguous.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from localization.geometry.descriptor import triangle_descriptors_from_polygon
from localization.database.patches import patch_owns_centroid
from localization.io.bounds import pixel_offset_to_meters

CENTER = 250.0


def dedup_points(pts, grid=10.0):
    if len(pts) == 0:
        return np.zeros((0, 2), np.float64)
    pts = np.asarray(pts, np.float64)
    keys = np.round(pts / grid).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def context_feats(tri_cents, bldg_pts, R, nd, na):
    """(T, nd*na) log-polar histogram of building centroids within R of each triangle."""
    T = tri_cents.shape[0]
    feats = np.zeros((T, nd * na), np.float32)
    if bldg_pts.shape[0] == 0 or T == 0:
        return feats
    tree = cKDTree(bldg_pts)
    logR = math.log1p(R)
    nbrs = tree.query_ball_point(tri_cents, R)
    for i, nb in enumerate(nbrs):
        if not nb:
            continue
        rel = bldg_pts[nb] - tri_cents[i]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        m = dist > 1e-6
        rel, dist = rel[m], dist[m]
        if dist.size == 0:
            continue
        ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
        dbin = np.clip((np.log1p(dist) / logR * nd).astype(int), 0, nd - 1)
        abin = np.clip((ang / 360.0 * na).astype(int), 0, na - 1)
        np.add.at(feats[i], dbin * na + abin, 1.0)
    return feats


def load_satellite(site_dir, max_depth, patch_size, stride, dedup_cell):
    descs, cents, bcents = [], [], []
    with gzip.open(site_dir / "satellite_polygons.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tx, ty = float(rec["tl"][0]), float(rec["tl"][1])
            for poly in rec["polygons"]:
                a = np.asarray(poly, np.float64)
                if a.shape[0] >= 3:
                    bcents.append(a.mean(axis=0) + [tx, ty])
                d, c = triangle_descriptors_from_polygon(poly, max_depth=int(max_depth))
                if d.shape[0] == 0:
                    continue
                c = c.copy(); c[:, 0] += tx; c[:, 1] += ty
                keep = np.fromiter(
                    (patch_owns_centroid((cx, cy), (tx, ty), patch_size, stride, cell_size=dedup_cell)
                     for cx, cy in c), dtype=bool, count=c.shape[0])
                if keep.any():
                    descs.append(d[keep]); cents.append(c[keep])
    if not descs:
        return np.zeros((0, 5), np.float32), np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float64)
    return np.vstack(descs), np.vstack(cents), dedup_points(bcents, grid=10.0)


def load_uav_queries(site_dir, max_depth, limit):
    import csv
    gt = {}
    with open(site_dir / "records_main.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("gt_px_x") not in (None, "") and r.get("gt_px_y") not in (None, ""):
                try:
                    gt[str(r["image"])] = (float(r["gt_px_x"]), float(r["gt_px_y"]))
                except ValueError:
                    pass
    out = []
    for jf in sorted((site_dir / "uav_polygons").glob("*.json")):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        name = rec.get("image", jf.stem)
        if name not in gt:
            continue
        descs, cents, bcents = [], [], []
        for poly in rec["polygons"]:
            a = np.asarray(poly, np.float64)
            if a.shape[0] >= 3:
                bcents.append(a.mean(axis=0))
            d, c = triangle_descriptors_from_polygon(poly, max_depth=int(max_depth))
            if d.shape[0]:
                descs.append(d); cents.append(c)
        if not descs:
            continue
        out.append((name, np.vstack(descs), np.vstack(cents),
                    np.asarray(bcents, np.float64), gt[name]))
        if limit and len(out) >= limit:
            break
    return out


def run(site, args):
    sd = Path(args.out) / site
    meta = json.loads((sd / "site_meta.json").read_text())
    bounds, sat_w, sat_h = meta["bounds"], meta["sat_width"], meta["sat_height"]
    use_ctx = not args.no_context
    print(f"\n===== site {site} | context={use_ctx} R={args.ctx_r} nd={args.ctx_nd} na={args.ctx_na} "
          f"K={args.k} bin={args.bin} =====", flush=True)

    sat_desc, sat_cent, sat_bldg = load_satellite(sd, args.max_depth, args.patch_size, args.stride, args.dedup_cell)
    print(f"[load] sat triangles={sat_desc.shape[0]} buildings={sat_bldg.shape[0]}", flush=True)
    queries = load_uav_queries(sd, args.max_depth, args.limit)
    print(f"[load] uav queries={len(queries)}", flush=True)
    if sat_desc.shape[0] == 0 or not queries:
        print("[skip] nothing"); return

    if use_ctx:
        sat_ctx = context_feats(sat_cent, sat_bldg, args.ctx_r, args.ctx_nd, args.ctx_na)
        sat_full = np.hstack([sat_desc, sat_ctx])
    else:
        sat_full = sat_desc
    # z-score every dim on the satellite set (balances angle block vs context block)
    mu = sat_full.mean(axis=0); sdv = sat_full.std(axis=0); sdv[sdv < 1e-6] = 1.0
    sat_n = (sat_full - mu) / sdv
    desc_tree = cKDTree(sat_n)
    cent_tree = cKDTree(sat_cent)

    a1_topk, hough_err, n = [], [], 0
    K = int(args.k)
    for name, d, c, ub, (gx, gy) in queries:
        if use_ctx:
            uctx = context_feats(c, ub, args.ctx_r, args.ctx_nd, args.ctx_na)
            df = np.hstack([d, uctx])
        else:
            df = d
        dn = (df - mu) / sdv
        exp = np.column_stack([gx + (c[:, 0] - CENTER), gy + (c[:, 1] - CENTER)])
        dist_k, idx_k = desc_tree.query(dn, k=K)
        if K == 1:
            idx_k = idx_k[:, None]
        n += 1
        for i in range(d.shape[0]):
            near = cent_tree.query_ball_point(exp[i], args.radius)
            if near:
                a1_topk.append(len(set(near).intersection(idx_k[i].tolist())) > 0)
        votes = []
        for i in range(d.shape[0]):
            for j in idx_k[i]:
                votes.append((sat_cent[j, 0] - (c[i, 0] - CENTER), sat_cent[j, 1] - (c[i, 1] - CENTER)))
        votes = np.asarray(votes)
        keys = np.floor(votes / args.bin).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        pred = votes[inv == int(counts.argmax())].mean(axis=0)
        hough_err.append(pixel_offset_to_meters(pred[0] - gx, pred[1] - gy, bounds, sat_w, sat_h)["distance_m"])

    he = np.asarray(hough_err)
    pct = lambda x: 100.0 * float(np.mean(x)) if len(x) else float("nan")
    print(f"[A1] geometrically-correct retrieved in top-{K}: {pct(a1_topk):.1f}%  (n_tri={len(a1_topk)})", flush=True)
    print(f"[A3] Hough error (m): mean={he.mean():.1f} median={np.median(he):.1f} "
          f"p25/p75={np.percentile(he,25):.0f}/{np.percentile(he,75):.0f}", flush=True)
    print(f"[A3] error <100m={pct(he<100):.1f}%  <50m={pct(he<50):.1f}%  <30m={pct(he<30):.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--bin", type=float, default=200.0)
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--ctx-r", type=float, default=250.0, dest="ctx_r")
    ap.add_argument("--ctx-nd", type=int, default=3, dest="ctx_nd")
    ap.add_argument("--ctx-na", type=int, default=8, dest="ctx_na")
    ap.add_argument("--no-context", action="store_true", dest="no_context")
    ap.add_argument("--max-depth", type=int, default=4, dest="max_depth")
    ap.add_argument("--patch-size", type=int, default=500, dest="patch_size")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--dedup-cell", type=float, default=250.0, dest="dedup_cell")
    args = ap.parse_args()
    for s in args.sites:
        run(s, args)


if __name__ == "__main__":
    main()
