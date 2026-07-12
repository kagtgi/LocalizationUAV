#!/usr/bin/env python
"""Ekeland-Constellation Descriptor (ECD) discriminability probe. CPU-only, on
cached polygons -- no GPU / rebuild. Keeps the Ekeland free-cone angle as the
per-building shape signature and tests three descriptors on the SAME building
candidate pool, isolating the marginal value of the angle on top of arrangement:

  mode 'ekeland' : per-building Ekeland-angle summary only  (no arrangement)
  mode 'arrange' : k-NN neighbourhood geometry only         (CFBVM-style, shape-blind)
  mode 'ecd'     : arrangement + Ekeland-angle node attributes (ours)

Metric: truth-anchored building retrieval @top-K -- for each UAV building, is its
geometrically-correct satellite building (nearest sat centroid to the GT-mapped
UAV centroid) among the top-K nearest descriptors? Same GT logic as phase_a_diag.
"""
from __future__ import annotations
import argparse, gzip, json, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from localization.geometry.descriptor import triangle_descriptors_from_polygon
from localization.database.patches import patch_owns_centroid

CENTER = 250.0


def poly_geom(poly):
    p = np.asarray(poly, float)
    if p.shape[0] < 3:
        return None
    x, y = p[:, 0], p[:, 1]
    A = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    per = float(np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T).sum())
    cx, cy = float(x.mean()), float(y.mean())
    return cx, cy, A, per


def building_feat(poly, max_depth):
    """Per-building feature: centroid + Ekeland-angle shape summary + size."""
    g = poly_geom(poly)
    if g is None:
        return None
    cx, cy, A, per = g
    d, _ = triangle_descriptors_from_polygon(poly, max_depth=max_depth)  # cols: a1,a2,e1,e2,e3
    if d.shape[0] == 0:
        min_e = mean_e = 180.0; a1 = a2 = 90.0; ntri = 0
    else:
        ek = d[:, 2:5]
        min_e = float(ek.min()); mean_e = float(ek.mean())
        a1 = float(d[:, 0].mean()); a2 = float(d[:, 1].mean()); ntri = int(d.shape[0])
    # ekeland shape signature (rotation/scale invariant) + calibrated size
    shape = np.array([min_e, mean_e, a1, a2, math.log1p(ntri)], np.float32)
    size = np.array([math.log1p(A), math.log1p(per)], np.float32)
    return {"c": (cx, cy), "shape": shape, "size": size}


def load_sat_buildings(site_dir, max_depth, patch_size, stride, dedup_cell):
    # cache is keyed by max_depth (features depend on it!); the original
    # unkeyed "ecd_sat_cache.npz" is the depth=4 cache (session default).
    cache = site_dir / (f"ecd_sat_cache_d{max_depth}.npz" if int(max_depth) != 4
                         else "ecd_sat_cache.npz")
    if cache.exists():
        z = np.load(cache)
        cents = z["cents"]; shapes = z["shapes"]; sizes = z["sizes"]
        feats = [{"c": (float(cents[i, 0]), float(cents[i, 1])),
                  "shape": shapes[i], "size": sizes[i]} for i in range(cents.shape[0])]
        print(f"[load] sat buildings from cache: {cents.shape[0]}", flush=True)
        return feats, cents
    feats, cents = [], []
    n_seen = n_owned = 0
    with gzip.open(site_dir / "satellite_polygons.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tlx, tly = float(rec["tl"][0]), float(rec["tl"][1])
            for poly in rec["polygons"]:
                n_seen += 1
                g = poly_geom(poly)                       # cheap centroid, no triangulation
                if g is None:
                    continue
                gx, gy = g[0] + tlx, g[1] + tly
                # dedup BEFORE the expensive triangulation (patches overlap ~16x)
                if not patch_owns_centroid((gx, gy), (tlx, tly), patch_size, stride, cell_size=dedup_cell):
                    continue
                bf = building_feat(poly, max_depth)        # triangulate only owned buildings
                if bf is None:
                    continue
                bf["c"] = (gx, gy)
                feats.append(bf); cents.append((gx, gy))
                n_owned += 1
    print(f"[load] sat polygons seen={n_seen} owned(deduped)={n_owned}", flush=True)
    cents = np.asarray(cents, float) if cents else np.zeros((0, 2))
    if feats:
        np.savez(cache, cents=cents,
                 shapes=np.vstack([f["shape"] for f in feats]),
                 sizes=np.vstack([f["size"] for f in feats]))
    return feats, cents


def load_uav_buildings(site_dir, max_depth, limit):
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
        feats, cents = [], []
        for poly in rec["polygons"]:
            bf = building_feat(poly, max_depth)
            if bf is None:
                continue
            feats.append(bf); cents.append(bf["c"])
        if feats:
            out.append((name, feats, np.asarray(cents, float), gt[name]))
        if limit and len(out) >= limit:
            break
    return out


def descriptors(feats, cents, k, mode):
    """Build (N,D) descriptor matrix over buildings in one coordinate frame.

    Fixed per-mode dimension (robust to isolated buildings / tiny frames):
      ekeland : 7                      (anchor Ekeland-shape + size only)
      arrange : 3k                     (k neighbours x [log-dist, sin, cos]; shape-blind)
      ecd     : 7 + 6k                 (anchor + k neighbours x [log-dist,sin,cos, min_e,mean_e,logA])
    """
    n = len(feats)
    per = 3 if mode == "arrange" else 6
    anchor_dim = 0 if mode == "arrange" else 7
    D = anchor_dim + (0 if mode == "ekeland" else k * per)
    if n == 0:
        return np.zeros((0, max(D, 1)))
    tree = cKDTree(cents)
    kk = min(k, max(1, n - 1))
    rows = []
    for i in range(n):
        f = feats[i]
        anchor = np.concatenate([f["shape"], f["size"]]) if anchor_dim else np.zeros(0, np.float32)
        if mode == "ekeland":
            rows.append(anchor.astype(np.float32)); continue
        dist, idx = tree.query(cents[i], k=min(kk + 1, n))
        dist = np.atleast_1d(dist); idx = np.atleast_1d(idx)
        pairs = [(float(d), int(j)) for d, j in zip(dist, idx)
                 if int(j) != i and int(j) < n and np.isfinite(d)][:kk]
        block = []
        if pairs:
            nn = pairs[0][1]
            theta0 = math.atan2(cents[nn][1] - cents[i][1], cents[nn][0] - cents[i][0])
            items = []
            for d, j in pairs:
                v = cents[j] - cents[i]
                bear = math.atan2(v[1], v[0]) - theta0
                items.append((math.log1p(d), math.sin(bear), math.cos(bear), feats[j]))
            items.sort(key=lambda t: math.atan2(t[1], t[2]))     # order-invariant by canonical bearing
            for logd, sb, cb, fj in items:
                if mode == "arrange":
                    block += [logd, sb, cb]                       # geometry only (shape-blind, CFBVM-style)
                else:
                    block += [logd, sb, cb, float(fj["shape"][0]), float(fj["shape"][1]), float(fj["size"][0])]
        need = k * per - len(block)
        if need > 0:
            block += [0.0] * need
        block = np.asarray(block[:k * per], np.float32)
        rows.append(block if mode == "arrange" else np.concatenate([anchor, block]).astype(np.float32))
    return np.vstack(rows)


def zfit(m):
    mu = m.mean(0); sd = m.std(0); sd[sd < 1e-6] = 1.0
    return mu, sd


def run(site, args):
    sd_dir = Path(args.out) / site
    sfeat, scent = load_sat_buildings(sd_dir, args.max_depth, args.patch_size, args.stride, args.dedup_cell)
    queries = load_uav_buildings(sd_dir, args.max_depth, args.limit)
    print(f"\n===== site {site}: {len(sfeat)} sat buildings, {len(queries)} UAV frames  "
          f"(neighbourhood knn={args.knn}) =====", flush=True)
    if not sfeat or not queries:
        print("[skip] missing cache"); return None
    scent_tree = cKDTree(scent)
    res = {}
    for mode in ["ekeland", "arrange", "ecd"]:
        sat_desc = descriptors(sfeat, scent, args.knn, mode)
        mu, sd = zfit(sat_desc)
        sat_tree = cKDTree((sat_desc - mu) / sd)
        topk, top100 = args.topk, args.top100
        hitk = hit100 = tot = 0
        for name, ufeat, ucent, (gx, gy) in queries:
            ud = descriptors(ufeat, ucent, args.knn, mode)
            udn = (ud - mu) / sd
            for i in range(len(ufeat)):
                ex = gx + (ucent[i, 0] - CENTER); ey = gy + (ucent[i, 1] - CENTER)
                near = scent_tree.query_ball_point([ex, ey], args.radius)
                if not near:
                    continue
                near = set(int(j) for j in near)
                _, idx = sat_tree.query(udn[i], k=min(top100, sat_desc.shape[0]))
                idx = [int(j) for j in np.atleast_1d(idx)]
                tot += 1
                if near & set(idx[:topk]):
                    hitk += 1
                if near & set(idx[:top100]):
                    hit100 += 1
        rk = 100.0 * hitk / tot if tot else float("nan")
        r100 = 100.0 * hit100 / tot if tot else float("nan")
        res[mode] = {f"retrieval_top{topk}": round(rk, 2),
                     f"retrieval_top{top100}": round(r100, 2),
                     "scored": tot, "dim": int(sat_desc.shape[1])}
        print(f"[{mode:8s}] dim={sat_desc.shape[1]:3d}  retrieval@top-{topk}={rk:.2f}%  "
              f"@top-{top100}={r100:.2f}%  ({tot} bldg-queries)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--knn", type=int, default=8, help="constellation neighbourhood size")
    ap.add_argument("--topk", type=int, default=20, help="retrieval top-K (primary)")
    ap.add_argument("--top100", type=int, default=100)
    ap.add_argument("--radius", type=float, default=30.0, help="GT match radius px")
    ap.add_argument("--max-depth", type=int, default=4, dest="max_depth")
    ap.add_argument("--patch-size", type=int, default=500, dest="patch_size")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--dedup-cell", type=float, default=250.0, dest="dedup_cell")
    args = ap.parse_args()
    all_res = {}
    for s in args.sites:
        r = run(s, args)
        if r:
            all_res[s] = r
    Path("outputs/paper").mkdir(parents=True, exist_ok=True)
    Path("outputs/paper/ecd_probe.json").write_text(json.dumps(all_res, indent=2))
    print("\n[ecd] wrote outputs/paper/ecd_probe.json", flush=True)


if __name__ == "__main__":
    main()
