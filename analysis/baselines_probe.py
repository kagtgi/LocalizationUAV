#!/usr/bin/env python
"""Real head-to-head baselines (review point 4: "no real head-to-head baseline
- the CFBVM-style descriptor is the authors' own reconstruction, not CFBVM").
Same cells, same candidate pool, same protocol as area_ekeland_probe2.py's
plain/ekeland modes -- just two more cell descriptors:

  mode 'brm'         : Building Ratio Map (Choi & Myung 2020) - building-area
                       coverage ratio in a small sub-grid within each cell
                       (a spatial density-ratio map, no shape/layout info).
  mode 'cfbvm_rings' : CFBVM (Ouyang et al. 2024) - concentric-ring building-
                       area summation around the cell centre (rotation-
                       invariant radial density profile, no per-building shape).

Reuses load_sat_buildings (ecd_probe.py) and the identical cell/GT/retrieval
harness as area_ekeland_probe2.py so results drop directly into the same table.
CPU-only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from ecd_probe import load_sat_buildings, building_feat, CENTER
from localization.io.bounds import pixel_offset_to_meters

N_RING = 6     # CFBVM-style concentric rings
N_GRID = 4     # BRM-style sub-grid side (4x4 = 16 sub-cells)


def brm_descriptor(c, areas, cell_x, cell_y, half):
    """Sub-grid building-area coverage ratio (Building Ratio Map)."""
    grid = np.zeros((N_GRID, N_GRID))
    sub = 2 * half / N_GRID
    if c.shape[0] == 0:
        return grid.flatten()
    for (px, py), a in zip(c, areas):
        gx = int(np.clip((px - (cell_x - half)) / sub, 0, N_GRID - 1))
        gy = int(np.clip((py - (cell_y - half)) / sub, 0, N_GRID - 1))
        grid[gy, gx] += a
    sub_area = sub * sub
    return (grid / sub_area).flatten()  # coverage ratio per sub-cell


def cfbvm_ring_descriptor(c, areas, cell_x, cell_y, half):
    """Concentric-ring building-area summation (CFBVM shape vector)."""
    if c.shape[0] == 0:
        return np.zeros(N_RING)
    d = np.hypot(c[:, 0] - cell_x, c[:, 1] - cell_y)
    edges = np.linspace(0, half * 1.4143, N_RING + 1)
    ring_area = np.zeros(N_RING)
    for i in range(N_RING):
        m = (d >= edges[i]) & (d < edges[i + 1])
        ring_area[i] = areas[m].sum()
        ann_area = np.pi * (edges[i + 1] ** 2 - edges[i] ** 2)
        ring_area[i] = ring_area[i] / max(ann_area, 1.0)  # normalize by annulus area
    return ring_area


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
        C, A = [], []
        for poly in rec["polygons"]:
            bf = building_feat(poly, max_depth)
            if bf is None:
                continue
            C.append(bf["c"]); A.append(float(np.expm1(bf["size"][0])))  # size[0]=log1p(area)
        if C:
            out.append((name, np.asarray(C, float), np.asarray(A, float), gt[name]))
        if limit and len(out) >= limit:
            break
    return out


def run(site, args):
    sd = Path(args.out) / site
    feats, cents = load_sat_buildings(sd, args.max_depth, args.patch_size, args.stride, args.dedup_cell)
    if not feats:
        print(f"[skip] {site}"); return None
    areas = np.array([float(np.expm1(f["size"][0])) for f in feats])
    meta = json.loads((sd / "site_meta.json").read_text())
    bounds, sw, sh_ = meta["bounds"], meta["sat_width"], meta["sat_height"]
    G = args.cell; half = G / 2.0
    keys = np.floor(cents / G).astype(np.int64)
    cell_map = {}
    for i, k in enumerate(map(tuple, keys)):
        cell_map.setdefault(k, []).append(i)
    cell_keys = [k for k, v in cell_map.items() if len(v) >= args.min_bldg]
    cell_ctr = np.array([[(k[0] + 0.5) * G, (k[1] + 0.5) * G] for k in cell_keys])
    ctr_tree = cKDTree(cell_ctr)
    frames = uav_frames(sd, args.max_depth, args.limit)
    print(f"\n===== site {site}: {cents.shape[0]} sat buildings -> {len(cell_keys)} cells "
          f"(G={G}px), {len(frames)} UAV frames =====", flush=True)
    res = {}
    for mode in ["brm", "cfbvm_rings"]:
        sat_desc = []
        for k, ctr in zip(cell_keys, cell_ctr):
            m = cell_map[k]
            c, a = cents[m], areas[m]
            if mode == "brm":
                sat_desc.append(brm_descriptor(c, a, ctr[0], ctr[1], half))
            else:
                sat_desc.append(cfbvm_ring_descriptor(c, a, ctr[0], ctr[1], half))
        sat_desc = np.vstack(sat_desc)
        mu = sat_desc.mean(0); sd_ = sat_desc.std(0); sd_[sd_ < 1e-6] = 1.0
        tree = cKDTree((sat_desc - mu) / sd_)
        topk, top100 = args.topk, args.top100
        hitk = hit100 = 0; errs = []; tot = 0
        frame_rows = []
        for name, C, A, (gx, gy) in frames:
            if mode == "brm":
                ud = brm_descriptor(C, A, CENTER, CENTER, half)
            else:
                ud = cfbvm_ring_descriptor(C, A, CENTER, CENTER, half)
            ud = (ud - mu) / sd_
            kq = min(top100, len(cell_keys))
            _, idx = tree.query(ud, k=kq)
            idx = [int(j) for j in np.atleast_1d(idx)]
            correct = set(int(j) for j in ctr_tree.query_ball_point([gx, gy], args.radius))
            if not correct:
                continue
            tot += 1
            hk = bool(correct & set(idx[:topk])); h100 = bool(correct & set(idx[:top100]))
            hitk += int(hk); hit100 += int(h100)
            pc = cell_ctr[idx[0]]
            err = pixel_offset_to_meters(pc[0] - gx, pc[1] - gy, bounds, sw, sh_)["distance_m"]
            errs.append(err)
            frame_rows.append({"name": name, f"hit_top{topk}": hk, f"hit_top{top100}": h100, "err_m": err})
        errs = np.asarray(errs)
        res[mode] = {f"retrieval_top{topk}_pct": round(100.0 * hitk / tot, 2) if tot else None,
                     f"retrieval_top{top100}_pct": round(100.0 * hit100 / tot, 2) if tot else None,
                     "median_coarse_err_m": round(float(np.median(errs)), 1) if len(errs) else None,
                     "scored_frames": tot, "n_cells": len(cell_keys), "dim": int(sat_desc.shape[1])}
        r = res[mode]
        print(f"[{mode:12s}] dim={r['dim']:2d} cells={r['n_cells']:4d}  "
              f"retrieval@top-{topk}={r[f'retrieval_top{topk}_pct']}%  @top-{top100}={r[f'retrieval_top{top100}_pct']}%  "
              f"median_err={r['median_coarse_err_m']}m  ({tot} frames)", flush=True)
        if args.dump_per_frame:
            out_pf = sd / "baselines_per_frame.json"
            existing = json.loads(out_pf.read_text()) if out_pf.exists() else {}
            existing[mode] = frame_rows
            out_pf.write_text(json.dumps(existing, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--cell", type=float, default=500.0)
    ap.add_argument("--min-bldg", type=int, default=3, dest="min_bldg")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--top100", type=int, default=50)
    ap.add_argument("--radius", type=float, default=250.0)
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
    Path("outputs/paper/baselines.json").write_text(json.dumps(allr, indent=2))
    print("\n[baselines] wrote outputs/paper/baselines.json", flush=True)


if __name__ == "__main__":
    main()
