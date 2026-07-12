#!/usr/bin/env python
"""Phase A - BUILDING-level constellation matching (CPU-only, cached data).

Instead of matching individual triangles (480k, highly ambiguous), match whole
buildings (~45k, more distinctive). Each building carries a shape descriptor
(log-area, log-perimeter, vertex count, compactness, bbox aspect, extent) plus a
log-polar constellation context (histogram of neighbouring building centroids).
Retrieval reported at top-100. This is the strongest form of the descriptor-
matching paradigm; if even this leaves retrieval near 0%, the paradigm is the
ceiling.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from localization.io.bounds import pixel_offset_to_meters

CENTER = 250.0


def shape_feats(poly):
    a = np.asarray(poly, np.float64)
    n = a.shape[0]
    x, y = a[:, 0], a[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    per = float(np.sum(np.hypot(x - np.roll(x, -1), y - np.roll(y, -1))))
    comp = 4 * math.pi * area / (per * per) if per > 1e-6 else 0.0
    w = float(x.max() - x.min()); h = float(y.max() - y.min())
    asp = max(w, h) / max(min(w, h), 1.0)
    ext = area / max(w * h, 1.0)
    feat = np.array([math.log1p(area), math.log1p(per), float(n), comp, asp, ext], np.float32)
    cen = np.array([x.mean(), y.mean()], np.float64)
    return cen, feat, float(area)


def context_feats(cents, landmarks, R, nd, na):
    T = cents.shape[0]
    out = np.zeros((T, nd * na), np.float32)
    if landmarks.shape[0] == 0 or T == 0:
        return out
    tree = cKDTree(landmarks)
    logR = math.log1p(R)
    for i, nb in enumerate(tree.query_ball_point(cents, R)):
        if not nb:
            continue
        rel = landmarks[nb] - cents[i]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        m = dist > 1e-6
        rel, dist = rel[m], dist[m]
        if dist.size == 0:
            continue
        ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
        db = np.clip((np.log1p(dist) / logR * nd).astype(int), 0, nd - 1)
        ab = np.clip((ang / 360.0 * na).astype(int), 0, na - 1)
        np.add.at(out[i], db * na + ab, 1.0)
    return out


def dedup_buildings(items, grid=10.0):
    best = {}
    for cen, feat, area in items:
        key = (int(round(cen[0] / grid)), int(round(cen[1] / grid)))
        if key not in best or area > best[key][2]:
            best[key] = (cen, feat, area)
    cens = np.array([v[0] for v in best.values()], np.float64)
    feats = np.array([v[1] for v in best.values()], np.float32)
    return cens, feats


def load_satellite(site_dir):
    items = []
    with gzip.open(site_dir / "satellite_polygons.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tx, ty = float(rec["tl"][0]), float(rec["tl"][1])
            for poly in rec["polygons"]:
                if len(poly) < 3:
                    continue
                cen, feat, area = shape_feats(poly)
                items.append((cen + [tx, ty], feat, area))
    return dedup_buildings(items, grid=10.0)


def load_uav(site_dir, limit):
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
        cens, feats = [], []
        for poly in rec["polygons"]:
            if len(poly) < 3:
                continue
            cen, feat, _ = shape_feats(poly)
            cens.append(cen); feats.append(feat)
        if not cens:
            continue
        out.append((name, np.asarray(cens, np.float64), np.asarray(feats, np.float32), gt[name]))
        if limit and len(out) >= limit:
            break
    return out


def run(site, args):
    sd = Path(args.out) / site
    meta = json.loads((sd / "site_meta.json").read_text())
    bounds, sat_w, sat_h = meta["bounds"], meta["sat_width"], meta["sat_height"]
    print(f"\n===== site {site} BUILDING-level | context={not args.no_context} R={args.ctx_r} "
          f"K={args.k} bin={args.bin} radius={args.radius} =====", flush=True)

    sat_cen, sat_shape = load_satellite(sd)
    print(f"[load] sat buildings={sat_cen.shape[0]}", flush=True)
    queries = load_uav(sd, args.limit)
    print(f"[load] uav queries={len(queries)}", flush=True)
    if sat_cen.shape[0] == 0 or not queries:
        print("[skip]"); return

    if args.no_context:
        sat_full = sat_shape
    else:
        sat_full = np.hstack([sat_shape, context_feats(sat_cen, sat_cen, args.ctx_r, args.ctx_nd, args.ctx_na)])
    mu = sat_full.mean(0); sdv = sat_full.std(0); sdv[sdv < 1e-6] = 1.0
    sat_n = (sat_full - mu) / sdv
    desc_tree = cKDTree(sat_n)
    cen_tree = cKDTree(sat_cen)

    K = int(args.k)
    a1, hough = [], []
    for name, uc, ufeat, (gx, gy) in queries:
        if args.no_context:
            uf = ufeat
        else:
            uf = np.hstack([ufeat, context_feats(uc, uc, args.ctx_r, args.ctx_nd, args.ctx_na)])
        un = (uf - mu) / sdv
        exp = np.column_stack([gx + (uc[:, 0] - CENTER), gy + (uc[:, 1] - CENTER)])
        kk = min(K, sat_cen.shape[0])
        _, idx = desc_tree.query(un, k=kk)
        if kk == 1:
            idx = idx[:, None]
        for i in range(uc.shape[0]):
            near = cen_tree.query_ball_point(exp[i], args.radius)
            if near:
                a1.append(len(set(near).intersection(idx[i].tolist())) > 0)
        votes = []
        for i in range(uc.shape[0]):
            for j in idx[i]:
                votes.append((sat_cen[j, 0] - (uc[i, 0] - CENTER), sat_cen[j, 1] - (uc[i, 1] - CENTER)))
        votes = np.asarray(votes)
        keys = np.floor(votes / args.bin).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        pred = votes[inv == int(counts.argmax())].mean(0)
        hough.append(pixel_offset_to_meters(pred[0] - gx, pred[1] - gy, bounds, sat_w, sat_h)["distance_m"])

    he = np.asarray(hough)
    pct = lambda x: 100.0 * float(np.mean(x)) if len(x) else float("nan")
    print(f"[A1] geometrically-correct building retrieved in top-{K}: {pct(a1):.1f}%  (n={len(a1)})", flush=True)
    print(f"[A3] Hough error (m): mean={he.mean():.1f} median={np.median(he):.1f} "
          f"p25/p75={np.percentile(he,25):.0f}/{np.percentile(he,75):.0f}", flush=True)
    print(f"[A3] error <100m={pct(he<100):.1f}%  <50m={pct(he<50):.1f}%  <30m={pct(he<30):.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--bin", type=float, default=200.0)
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--ctx-r", type=float, default=250.0, dest="ctx_r")
    ap.add_argument("--ctx-nd", type=int, default=3, dest="ctx_nd")
    ap.add_argument("--ctx-na", type=int, default=8, dest="ctx_na")
    ap.add_argument("--no-context", action="store_true", dest="no_context")
    args = ap.parse_args()
    for s in args.sites:
        run(s, args)


if __name__ == "__main__":
    main()
