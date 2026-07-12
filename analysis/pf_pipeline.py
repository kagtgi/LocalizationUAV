#!/usr/bin/env python
"""CFBVM-style sequential pipeline: our geometric descriptor as the matching stage,
per-frame Hough votes as a GMM observation, integrated over the flight trajectory
by a particle filter. CPU-only, on cached flight-01 query artifacts.

Stage 0 (--diag): does the true nadir accumulate consistent vote support across
consecutive frames? (Decides whether any filter can converge.)
Stage 1 (default): run the particle filter, report trajectory-level error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from localization.database.kdtree import SatelliteDatabase
from localization.geometry.descriptor import triangle_descriptors_from_polygon
from localization.io.bounds import pixel_offset_to_meters

CENTER = 250.0


def frame_votes(db, polys, k):
    """Cloud of candidate nadir positions for one frame: N = sat_centroid - (uav_c - center)."""
    descs, cents = [], []
    for p in polys:
        d, c = triangle_descriptors_from_polygon(p, max_depth=4)
        if d.shape[0]:
            descs.append(d); cents.append(c)
    if not descs:
        return np.zeros((0, 2))
    desc = np.vstack(descs).astype(np.float32)
    cent = np.vstack(cents)
    _, idx = db.query(desc, k=min(k, db.size), p=1)
    if idx.ndim == 1:
        idx = idx[:, None]
    sat = db.centroids[idx.reshape(-1)]              # (M*k, 2)
    off = np.repeat(cent - CENTER, idx.shape[1], axis=0)  # (M*k, 2)
    return sat - off                                  # nadir votes


def load_sequence(sp_dir, limit):
    rec = pd.read_csv(sp_dir / "records_main.csv", dtype={"image": str})
    rec = rec[rec["gt_px_x"].notna() & rec["gt_px_y"].notna()].copy()
    rec = rec.sort_values("image").head(limit)
    frames = []
    for _, r in rec.iterrows():
        stem = Path(str(r["image"])).stem
        jf = sp_dir / "uav_polygons" / f"{stem}.json"
        if not jf.exists():
            continue
        polys = json.loads(jf.read_text())["polygons"]
        frames.append((str(r["image"]), polys, float(r["gt_px_x"]), float(r["gt_px_y"])))
    return frames


def diagnostic(db, frames, k, r_obs, bounds, sat_w, sat_h):
    print(f"[diag] {len(frames)} consecutive frames", flush=True)
    gts = np.array([[gx, gy] for _, _, gx, gy in frames])
    disp = np.linalg.norm(np.diff(gts, axis=0), axis=1)
    md = float(np.median(disp))
    print(f"[diag] inter-frame GT displacement: median={md:.0f}px "
          f"({md*pixel_offset_to_meters(md,0,bounds,sat_w,sat_h)['m_per_px_x']:.0f}m approx), max={disp.max():.0f}px", flush=True)
    support, ranks = [], []
    for name, polys, gx, gy in frames:
        v = frame_votes(db, polys, k)
        if v.shape[0] == 0:
            support.append(0); continue
        from scipy.spatial import cKDTree
        vt = cKDTree(v)
        n_gt = len(vt.query_ball_point([gx, gy], r_obs))     # votes near TRUE position
        support.append(n_gt)
        # rank of GT vote-density among a coarse grid of candidate positions
        gs = 500.0
        keys = np.floor(v / gs).astype(np.int64)
        _, counts = np.unique(keys, axis=0, return_counts=True)
        gt_cell = np.floor(np.array([gx, gy]) / gs).astype(np.int64)
        gt_count = int(np.sum((keys == gt_cell).all(axis=1)))
        ranks.append(1 + int(np.sum(counts > gt_count)) if gt_count > 0 else -1)
    support = np.array(support)
    got = support > 0
    print(f"[diag] frames with >=1 vote within {r_obs:.0f}px of TRUE pos: {100*got.mean():.1f}%", flush=True)
    print(f"[diag] GT-support votes/frame: median={np.median(support):.0f} mean={support.mean():.1f} max={support.max()}", flush=True)
    rr = np.array([r for r in ranks if r > 0])
    if rr.size:
        print(f"[diag] when GT supported, its 500px-cell vote rank: median={np.median(rr):.0f} "
              f"(<=10 in {100*np.mean(rr<=10):.0f}% of supported frames, <=100 in {100*np.mean(rr<=100):.0f}%)", flush=True)
    print(f"[diag] VERDICT: {'promising - true pos gets recurring support; PF can integrate' if got.mean()>0.2 else 'weak - true pos rarely supported; PF unlikely to converge'}", flush=True)


def particle_filter(db, frames, args, bounds, sat_w, sat_h):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    W, H = sat_w, sat_h
    N = args.particles
    px = rng.uniform(0, W, N); py = rng.uniform(0, H, N)
    w = np.full(N, 1.0 / N)
    r_obs = args.r_obs
    errs, conv = [], None
    for t, (name, polys, gx, gy) in enumerate(frames):
        # predict: random-walk motion (isotropic; magnitude = assumed drone speed)
        px = px + rng.normal(0, args.move_std, N)
        py = py + rng.normal(0, args.move_std, N)
        # a small fraction re-seeded globally to recover from divergence / global init
        nre = int(args.reseed * N)
        if nre:
            ridx = rng.choice(N, nre, replace=False)
            px[ridx] = rng.uniform(0, W, nre); py[ridx] = rng.uniform(0, H, nre)
        px = np.clip(px, 0, W); py = np.clip(py, 0, H)
        # update: observation likelihood = local vote density (GMM-ish, floored for credibility)
        v = frame_votes(db, polys, args.k)
        if v.shape[0] >= 3:
            vt = cKDTree(v)
            cnt = np.array(vt.query_ball_point(np.column_stack([px, py]), r_obs, return_length=True), dtype=float)
            lik = args.floor + cnt
            # credibility: if the frame's best density is weak/diffuse, soften the update
            if cnt.max() < args.cred_min:
                lik = args.floor + 0.2 * cnt
            w = w * lik
        s = w.sum()
        w = w / s if s > 0 else np.full(N, 1.0 / N)
        # estimate
        ex = float(np.sum(w * px)); ey = float(np.sum(w * py))
        err = pixel_offset_to_meters(ex - gx, ey - gy, bounds, sat_w, sat_h)["distance_m"]
        errs.append(err)
        if conv is None and err < args.conv_m:
            conv = t
        # resample if degenerate
        ess = 1.0 / np.sum(w ** 2)
        if ess < N / 2:
            c = np.cumsum(w); c[-1] = 1.0
            u = (rng.random() + np.arange(N)) / N
            idx = np.searchsorted(c, u)
            px, py = px[idx], py[idx]
            px += rng.normal(0, args.move_std * 0.5, N)
            py += rng.normal(0, args.move_std * 0.5, N)
            w = np.full(N, 1.0 / N)
    errs = np.array(errs)
    tail = errs[max(0, len(errs) - args.tail):]
    print(f"[pf] frames={len(errs)} particles={N} move_std={args.move_std} r_obs={r_obs}", flush=True)
    print(f"[pf] converged (<{args.conv_m}m) at frame {conv}" if conv is not None
          else f"[pf] never converged below {args.conv_m}m", flush=True)
    print(f"[pf] error over last {len(tail)} frames: median={np.median(tail):.1f}m mean={tail.mean():.1f}m "
          f"min={tail.min():.1f}m", flush=True)
    print(f"[pf] last-{len(tail)} frames <100m={100*np.mean(tail<100):.0f}% <50m={100*np.mean(tail<50):.0f}% "
          f"<30m={100*np.mean(tail<30):.0f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval"); ap.add_argument("--site", default="01")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--r-obs", type=float, default=150.0, dest="r_obs")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--particles", type=int, default=20000)
    ap.add_argument("--move-std", type=float, default=200.0, dest="move_std")
    ap.add_argument("--reseed", type=float, default=0.05)
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--cred-min", type=float, default=3.0, dest="cred_min")
    ap.add_argument("--conv-m", type=float, default=100.0, dest="conv_m")
    ap.add_argument("--tail", type=int, default=30)
    args = ap.parse_args()

    sp_dir = Path(args.out) / args.site
    meta = json.loads((sp_dir / "site_meta.json").read_text())
    bounds, sat_w, sat_h = meta["bounds"], meta["sat_width"], meta["sat_height"]
    db = SatelliteDatabase.load(str(sp_dir / "satellite_kdtree.npz"))
    frames = load_sequence(sp_dir, args.limit)
    print(f"[load] db={db.size} triangles, {len(frames)} frames with GT", flush=True)
    diagnostic(db, frames, args.k, args.r_obs, bounds, sat_w, sat_h)
    if not args.diag:
        particle_filter(db, frames, args, bounds, sat_w, sat_h)


if __name__ == "__main__":
    main()
