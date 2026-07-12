#!/usr/bin/env python
"""Phase A offline validation (CPU-only, no GPU rebuild).

Runs on the VM inside ~/LocalizationUAV against cached artifacts from an
earlier build+query run:
  outputs/eval/<site>/satellite_polygons.jsonl.gz   {"patch_id","tl":[x,y],"polygons":[...]}
  outputs/eval/<site>/uav_polygons/<stem>.json      {"image","polygons":[...]}  (500x500-local)
  outputs/eval/<site>/records_main.csv              gt_px_x, gt_px_y per image
  outputs/eval/<site>/site_meta.json                bounds, sat_width, sat_height

Measures, without touching Mask R-CNN or rebuilding any DB:
  A1  retrievability     - is a UAV triangle's geometrically-correct satellite
                           counterpart within its top-K descriptor neighbours?
  A2  coverage           - fraction of queries with any satellite building near GT
  A3  translation-Hough  - consensus position estimate + error vs GT (the metric
                           that must drop from ~3300 m to low hundreds to pass)

Flags let each idea be A/B tested cheaply:
  --canonicalize --spacing N   resample every polygon boundary at N-px arc length
                               before triangulation (Phase A4)
  --zscore                     standardise each descriptor dim by its satellite-set
                               std before matching (Phase A5)
  --k / --bin / --radius       sweep retrieval K, Hough bin px, A1 match radius px

Self-test (no VM data):  python phase_a_diag.py --selftest
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

CENTER = 250.0  # nadir point in the 500x500 preprocessed UAV frame


# --------------------------------------------------------------------------- geometry helpers

def resample_polygon(poly, spacing):
    """Resample a closed polygon boundary at fixed arc-length spacing.

    Makes two independently-segmented outlines of the same building produce
    comparable vertex sequences (hence comparable CDT triangles), which is the
    Phase A4 hypothesis. Falls back to the original polygon when it is too
    small to carry >=3 samples at this spacing.
    """
    pts = np.asarray(poly, dtype=float)
    if pts.shape[0] < 3:
        return poly
    closed = np.vstack([pts, pts[:1]])
    seg = np.diff(closed, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    perim = float(seglen.sum())
    if perim < spacing * 3:
        return poly
    n = max(3, int(round(perim / spacing)))
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    out = []
    for t in np.linspace(0.0, perim, n, endpoint=False):
        k = int(np.searchsorted(cum, t, side="right") - 1)
        k = min(max(k, 0), len(seg) - 1)
        f = (t - cum[k]) / max(seglen[k], 1e-9)
        out.append((closed[k] + f * seg[k]).tolist())
    return out


def polys_to_desc_cent(polygons, max_depth, offset_xy, canonicalize, spacing, include_size=False):
    """(descriptors (T,D), global centroids (T,2)) for a list of polygons.

    With include_size, appends [log(1+area), log(1+perimeter)] (px units, both
    branches share the calibrated px scale) so D=7 - the scale-carrying
    dimensions that test whether the pure-angle descriptor's low entropy is the
    retrieval ceiling.
    """
    descs, cents = [], []
    dim = 7 if include_size else 5
    ox, oy = float(offset_xy[0]), float(offset_xy[1])
    for poly in polygons:
        p = resample_polygon(poly, spacing) if canonicalize else poly
        if include_size:
            d, c, s = triangle_descriptors_from_polygon(p, max_depth=int(max_depth), include_size=True)
            if d.shape[0] == 0:
                continue
            extra = np.log1p(s.astype(np.float64))  # (T,2): log(1+area), log(1+perim)
            d = np.hstack([d, extra.astype(np.float32)])
        else:
            d, c = triangle_descriptors_from_polygon(p, max_depth=int(max_depth))
            if d.shape[0] == 0:
                continue
        c = c.copy()
        c[:, 0] += ox
        c[:, 1] += oy
        descs.append(d)
        cents.append(c)
    if not descs:
        return np.zeros((0, dim), np.float32), np.zeros((0, 2), np.float32)
    return np.vstack(descs), np.vstack(cents)


# --------------------------------------------------------------------------- loaders

def load_satellite(site_dir, max_depth, patch_size, stride, dedup_cell, canonicalize, spacing, include_size=False):
    descs, cents = [], []
    with gzip.open(site_dir / "satellite_polygons.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tl = (rec["tl"][0], rec["tl"][1])
            d, c = polys_to_desc_cent(rec["polygons"], max_depth, tl, canonicalize, spacing, include_size)
            if d.shape[0] == 0:
                continue
            keep = np.fromiter(
                (patch_owns_centroid((cx, cy), tl, patch_size, stride, cell_size=dedup_cell)
                 for cx, cy in c),
                dtype=bool, count=c.shape[0],
            )
            if keep.any():
                descs.append(d[keep])
                cents.append(c[keep])
    if not descs:
        return np.zeros((0, 5), np.float32), np.zeros((0, 2), np.float32)
    return np.vstack(descs), np.vstack(cents)


def load_uav_queries(site_dir, max_depth, canonicalize, spacing, limit, include_size=False):
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
    pdir = site_dir / "uav_polygons"
    for jf in sorted(pdir.glob("*.json")):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        name = rec.get("image", jf.stem)
        if name not in gt:
            continue
        d, c = polys_to_desc_cent(rec["polygons"], max_depth, (0.0, 0.0), canonicalize, spacing, include_size)
        if d.shape[0] == 0:
            continue
        out.append((name, d, c, gt[name]))
        if limit and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- metrics

def zscore_fit(desc):
    mu = desc.mean(axis=0)
    sd = desc.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def run_site(site, args):
    site_dir = Path(args.out) / site
    print(f"\n===== site {site}  (canonicalize={args.canonicalize} spacing={args.spacing} "
          f"zscore={args.zscore} size={args.size} K={args.k} bin={args.bin} radius={args.radius}) =====", flush=True)
    meta = json.loads((site_dir / "site_meta.json").read_text())
    bounds, sat_w, sat_h = meta["bounds"], meta["sat_width"], meta["sat_height"]

    sat_desc, sat_cent = load_satellite(
        site_dir, args.max_depth, args.patch_size, args.stride, args.dedup_cell,
        args.canonicalize, args.spacing, args.size)
    print(f"[load] satellite triangles: {sat_desc.shape[0]} (dim={sat_desc.shape[1] if sat_desc.shape[0] else '-'})", flush=True)
    queries = load_uav_queries(site_dir, args.max_depth, args.canonicalize, args.spacing, args.limit, args.size)
    print(f"[load] UAV queries with GT + descriptors: {len(queries)}", flush=True)
    if sat_desc.shape[0] == 0 or not queries:
        print("[skip] nothing to score")
        return

    dim = sat_desc.shape[1]
    if args.zscore:
        mu, sd = zscore_fit(sat_desc)
    else:
        mu, sd = np.zeros(dim, np.float32), np.ones(dim, np.float32)
    sat_desc_n = (sat_desc - mu) / sd
    desc_tree = cKDTree(sat_desc_n)
    cent_tree = cKDTree(sat_cent)

    a1_true, a1_topk, a2_cov = [], [], []
    hough_err, hough_hit = [], []
    n_scored = 0
    K = int(args.k)
    roi = float(args.roi_px) if args.roi_px else 0.0
    for name, d, c, (gx, gy) in queries:
        # A2 coverage: any satellite building within `cov_radius` px of GT?
        a2_cov.append(len(cent_tree.query_ball_point([gx, gy], args.cov_radius)) > 0)

        dn = (d - mu) / sd
        # expected satellite location of each UAV triangle if the UAV sits at GT
        exp = np.column_stack([gx + (c[:, 0] - CENTER), gy + (c[:, 1] - CENTER)])

        # Constrained-area (coarse-prior) mode: restrict the satellite candidate
        # set to a roi_px window around GT, then retrieve within it - mirrors how
        # constrained-area baselines (BRM ~5.7km2, abBRIEF ~1km2) operate. A
        # roi_px window >> target accuracy so sub-window error is real signal.
        if roi > 0:
            roi_ids = np.asarray(cent_tree.query_ball_point([gx, gy], roi), dtype=int)
            if roi_ids.size < K + 1:
                continue
            local_tree = cKDTree(sat_desc_n[roi_ids])
            dist_k, loc = local_tree.query(dn, k=K)
            if K == 1:
                dist_k = dist_k[:, None]; loc = loc[:, None]
            idx_k = roi_ids[loc]  # map local -> global indices
        else:
            dist_k, idx_k = desc_tree.query(dn, k=K)
            if K == 1:
                dist_k = dist_k[:, None]; idx_k = idx_k[:, None]
        n_scored += 1
        for i in range(d.shape[0]):
            near = cent_tree.query_ball_point(exp[i], args.radius)
            if not near:
                continue
            near = np.asarray(near)
            true_dd = float(np.abs(sat_desc_n[near] - dn[i]).sum(axis=1).min())
            a1_true.append(true_dd)
            # is any geometrically-correct triangle among this UAV triangle's top-K?
            a1_topk.append(len(set(near).intersection(idx_k[i].tolist())) > 0)

        # A3: translation-Hough. Each descriptor match casts a nadir vote
        # N = sat_centroid - (uav_local_centroid - center).
        votes = []
        for i in range(d.shape[0]):
            for j in idx_k[i]:
                votes.append((sat_cent[j, 0] - (c[i, 0] - CENTER),
                              sat_cent[j, 1] - (c[i, 1] - CENTER)))
        votes = np.asarray(votes)
        b = args.bin
        keys = np.floor(votes / b).astype(np.int64)
        uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        win = int(counts.argmax())
        inliers = votes[inv == win]
        pred = inliers.mean(axis=0)
        off = pixel_offset_to_meters(pred[0] - gx, pred[1] - gy, bounds, sat_w, sat_h)
        hough_err.append(off["distance_m"])
        hough_hit.append(abs(pred[0] - gx) <= b and abs(pred[1] - gy) <= b)

    def pct(x):
        return 100.0 * float(np.mean(x)) if len(x) else float("nan")

    he = np.asarray(hough_err)
    roi_note = f" | roi_px={roi:.0f}" if roi > 0 else " | full-map"
    print(f"[scored] {n_scored}/{len(queries)} queries scored{roi_note}", flush=True)
    print(f"[A2] coverage (>=1 sat building within {args.cov_radius}px of GT): {pct(a2_cov):.1f}%", flush=True)
    if a1_true:
        print(f"[A1] true-match descriptor L1: median={np.median(a1_true):.2f} mean={np.mean(a1_true):.2f}  "
              f"(n={len(a1_true)})", flush=True)
    print(f"[A1] geometrically-correct match retrieved in top-{K}: {pct(a1_topk):.1f}%", flush=True)
    if he.size:
        print(f"[A3] Hough position error (m): mean={he.mean():.1f} median={np.median(he):.1f}  "
              f"p25/p75={np.percentile(he,25):.0f}/{np.percentile(he,75):.0f}", flush=True)
        print(f"[A3] Hough peak within one bin of GT: {pct(hough_hit):.1f}%", flush=True)
        print(f"[A3] error <100m: {pct(he < 100):.1f}%  <50m: {pct(he < 50):.1f}%  "
              f"<30m: {pct(he < 30):.1f}%", flush=True)
    else:
        print("[A3] no queries scored (roi too small for K)", flush=True)


def selftest():
    """Synthetic check of the Hough + resampling logic (no VM data needed)."""
    rng = np.random.default_rng(0)
    # square resamples to ~ 4*side/spacing points, evenly spaced, closed
    sq = [[0, 0], [40, 0], [40, 40], [0, 40]]
    rs = resample_polygon(sq, 10.0)
    assert len(rs) >= 12, len(rs)
    per = sum(math.dist(rs[i], rs[(i + 1) % len(rs)]) for i in range(len(rs)))
    assert abs(per - 160.0) < 1.0, per
    # Hough: 10% inliers at a known nadir among 90% uniform noise -> recovered
    true_N = np.array([1234.0, 5678.0])
    inl = true_N + rng.normal(0, 2, size=(20, 2))
    noise = rng.uniform([0, 0], [9000, 26000], size=(180, 2))
    votes = np.vstack([inl, noise]); b = 50.0
    keys = np.floor(votes / b).astype(np.int64)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    pred = votes[inv == int(counts.argmax())].mean(axis=0)
    assert np.linalg.norm(pred - true_N) < 10, pred
    print("selftest OK: resampling + Hough consensus behave as expected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01", "11"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--bin", type=float, default=200.0)
    ap.add_argument("--radius", type=float, default=20.0, help="A1 geometric-match radius px")
    ap.add_argument("--cov-radius", type=float, default=500.0, dest="cov_radius")
    ap.add_argument("--canonicalize", action="store_true")
    ap.add_argument("--spacing", type=float, default=10.0)
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--size", action="store_true", help="append log(area),log(perim) dims (5D->7D)")
    ap.add_argument("--roi-px", type=float, default=0.0, dest="roi_px",
                    help="constrained-area mode: restrict satellite search to this radius (px) around GT (0=full map)")
    ap.add_argument("--max-depth", type=int, default=4, dest="max_depth")
    ap.add_argument("--patch-size", type=int, default=500, dest="patch_size")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--dedup-cell", type=float, default=250.0, dest="dedup_cell")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    for site in args.sites:
        run_site(site, args)


if __name__ == "__main__":
    main()
