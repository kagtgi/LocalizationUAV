#!/usr/bin/env python
"""Area-level Ekeland descriptor probe. Matches PATCHES (regions), not individual
buildings -- CFBVM-style area unit, robust to per-building segmentation noise --
with the Ekeland free-cone angle carried as a per-patch HISTOGRAM (the shape
signal CFBVM's density field lacks). CPU-only; reuses ecd_sat_cache.npz.

Ablation isolates the Ekeland contribution at area level:
  mode 'plain'   : density + layout + size histogram         (no Ekeland; CFBVM-like)
  mode 'ekeland' : + Ekeland-angle histograms (min_e, mean_e, a1)   (ours)

Metric: coarse-localization retrieval @top-K -- for each UAV frame, is the
map cell containing its GT position among the top-K nearest patch descriptors?
Plus median coarse-localization error (nearest-cell center vs GT), in meters.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from ecd_probe import load_sat_buildings, building_feat, CENTER
from localization.io.bounds import pixel_offset_to_meters

NB = 8


def _hist(vals, lo, hi):
    """NaN-safe normalized histogram: clip into range, normalize by count."""
    if vals.size == 0:
        return np.zeros(NB)
    h, _ = np.histogram(np.clip(vals, lo, hi), bins=NB, range=(lo, hi))
    s = h.sum()
    return h / s if s > 0 else np.zeros(NB)


def cell_descriptor(c, sh, sz, mode):
    """c:(m,2) centroids, sh:(m,5)[min_e,mean_e,a1,a2,logn], sz:(m,2)[logA,logP]."""
    n = c.shape[0]
    dens = np.log1p(n)
    if n > 1:
        rel = c - c.mean(0)
        spread = float(np.hypot(rel[:, 0], rel[:, 1]).mean())
        dd, _ = cKDTree(c).query(c, k=2)
        nnd = float(dd[:, 1].mean())
    else:
        spread = nnd = 0.0
    base = [dens, np.log1p(spread), np.log1p(nnd), *_hist(sz[:, 0], 0, 12)]
    if mode == "plain":
        v = np.asarray(base, np.float32)
    else:
        v = np.asarray([*base, *_hist(sh[:, 0], 90, 180), *_hist(sh[:, 1], 90, 180),
                        *_hist(sh[:, 2], 0, 180)], np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def uav_frames(site_dir, max_depth, limit):
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
        C, SH, SZ = [], [], []
        for poly in rec["polygons"]:
            bf = building_feat(poly, max_depth)
            if bf is None:
                continue
            C.append(bf["c"]); SH.append(bf["shape"]); SZ.append(bf["size"])
        if C:
            out.append((name, np.asarray(C, float), np.asarray(SH, float),
                        np.asarray(SZ, float), gt[name]))
        if limit and len(out) >= limit:
            break
    return out


def run(site, args):
    sd = Path(args.out) / site
    feats, cents = load_sat_buildings(sd, args.max_depth, args.patch_size, args.stride, args.dedup_cell)
    if not feats:
        print(f"[skip] {site}"); return None
    shapes = np.vstack([f["shape"] for f in feats])
    sizes = np.vstack([f["size"] for f in feats])
    meta = json.loads((sd / "site_meta.json").read_text())
    bounds, sw, sh_ = meta["bounds"], meta["sat_width"], meta["sat_height"]
    G = args.cell
    half = G / 2.0
    btree = cKDTree(cents)
    if args.sliding > 0:
        # overlapping candidate windows at fixed stride -> better UAV-frame alignment
        xs = np.arange(cents[:, 0].min(), cents[:, 0].max() + 1, args.sliding)
        ys = np.arange(cents[:, 1].min(), cents[:, 1].max() + 1, args.sliding)
        members, ctrs = [], []
        for x in xs:
            for y in ys:
                cand = btree.query_ball_point([x, y], half * 1.4143)
                ids = [i for i in cand if abs(cents[i, 0] - x) <= half and abs(cents[i, 1] - y) <= half]
                if len(ids) >= args.min_bldg:
                    members.append(ids); ctrs.append((x, y))
        cell_ctr = np.asarray(ctrs)
        member_list = members
    else:
        keys = np.floor(cents / G).astype(np.int64)
        cell_map = {}
        for i, k in enumerate(map(tuple, keys)):
            cell_map.setdefault(k, []).append(i)
        cell_keys = [k for k, v in cell_map.items() if len(v) >= args.min_bldg]
        cell_ctr = np.array([[(k[0] + 0.5) * G, (k[1] + 0.5) * G] for k in cell_keys])
        member_list = [cell_map[k] for k in cell_keys]
    ctr_tree = cKDTree(cell_ctr)
    frames = uav_frames(sd, args.max_depth, args.limit)
    print(f"\n===== site {site}: {cents.shape[0]} sat buildings -> {len(member_list)} cells "
          f"(G={G}px, sliding={args.sliding}), {len(frames)} UAV frames =====", flush=True)
    res = {}
    per_frame = {}
    for mode in ["plain", "ekeland"]:
        sat_desc = np.vstack([cell_descriptor(cents[m], shapes[m], sizes[m], mode) for m in member_list])
        mu = sat_desc.mean(0); sd_ = sat_desc.std(0); sd_[sd_ < 1e-6] = 1.0
        tree = cKDTree((sat_desc - mu) / sd_)
        topk, top100 = args.topk, args.top100
        hitk = hit100 = 0; errs = []; tot = 0
        frame_rows = []
        for name, C, SH, SZ, (gx, gy) in frames:
            ud = (cell_descriptor(C, SH, SZ, mode) - mu) / sd_
            kq = min(top100, len(member_list))
            _, idx = tree.query(ud, k=kq)
            idx = [int(j) for j in np.atleast_1d(idx)]
            # correct = any candidate window whose centre is within `radius` px of GT
            correct = set(int(j) for j in ctr_tree.query_ball_point([gx, gy], args.radius))
            if not correct:
                continue
            tot += 1
            hk = bool(correct & set(idx[:topk]))
            h100 = bool(correct & set(idx[:top100]))
            hitk += int(hk); hit100 += int(h100)
            pc = cell_ctr[idx[0]]
            err = pixel_offset_to_meters(pc[0] - gx, pc[1] - gy, bounds, sw, sh_)["distance_m"]
            errs.append(err)
            frame_rows.append({"name": name, f"hit_top{topk}": hk, f"hit_top{top100}": h100, "err_m": err})
        errs = np.asarray(errs)
        res[mode] = {f"retrieval_top{topk}_pct": round(100.0 * hitk / tot, 2) if tot else None,
                     f"retrieval_top{top100}_pct": round(100.0 * hit100 / tot, 2) if tot else None,
                     "median_coarse_err_m": round(float(np.median(errs)), 1) if len(errs) else None,
                     "scored_frames": tot, "n_cells": len(member_list), "dim": int(sat_desc.shape[1])}
        per_frame[mode] = frame_rows
        r = res[mode]
        print(f"[{mode:8s}] dim={r['dim']:2d} cells={r['n_cells']:4d}  "
              f"retrieval@top-{topk}={r[f'retrieval_top{topk}_pct']}%  @top-{top100}={r[f'retrieval_top{top100}_pct']}%  "
              f"median_err={r['median_coarse_err_m']}m  ({tot} frames)", flush=True)
    if args.dump_per_frame:
        out_pf = Path(args.out) / site / "area_per_frame.json"
        out_pf.write_text(json.dumps(per_frame, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--cell", type=float, default=500.0, help="window size px (~200m)")
    ap.add_argument("--sliding", type=float, default=0.0, help="if >0, overlapping windows at this stride px")
    ap.add_argument("--min-bldg", type=int, default=3, dest="min_bldg")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--top100", type=int, default=50)
    ap.add_argument("--radius", type=float, default=250.0, help="correct-window centre radius px")
    ap.add_argument("--max-depth", type=int, default=4, dest="max_depth")
    ap.add_argument("--patch-size", type=int, default=500, dest="patch_size")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--dedup-cell", type=float, default=250.0, dest="dedup_cell")
    ap.add_argument("--dump-per-frame", action="store_true", dest="dump_per_frame")
    args = ap.parse_args()
    allr = {}
    for s in args.sites:
        r = run(s, args)
        if r:
            allr[s] = r
    Path("outputs/paper").mkdir(parents=True, exist_ok=True)
    Path("outputs/paper/area_ekeland.json").write_text(json.dumps(allr, indent=2))
    print("\n[area] wrote outputs/paper/area_ekeland.json", flush=True)


if __name__ == "__main__":
    main()
