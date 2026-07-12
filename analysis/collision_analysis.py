#!/usr/bin/env python
"""Collision/entropy analysis replacing the trivial propositions (review point 2):
the paper's theorems certify coordinates-differ => vectors-differ, which is
irrelevant; what matters is IMPOSTOR DENSITY at the true-match distance. This
script measures that directly and derives one useful, provable statement:
  E[rank of true match] ~= 1 + N * P(random impostor within d_true)
which -> retrieval@K ~ 0 whenever N*P(...) >> K. Also reports the descriptor's
effective dimensionality via PCA (formalizing "~3 effective dims, not 5").

Reuses the cached satellite_descriptors.csv (5-D: alpha1,alpha2,e1,e2,e3) and the
same UAV-query / true-match-radius methodology as phase_a_diag.py. CPU-only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FIGDIR = Path("outputs/paper")
DESC_COLS = ["alpha1", "alpha2", "e1", "e2", "e3"]
CENTER = 250.0


def load_satellite_desc(site_dir):
    df = pd.read_csv(site_dir / "satellite_descriptors.csv")
    desc = df[DESC_COLS].to_numpy(dtype=np.float64)
    cent = df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    return desc, cent


def load_uav_queries(site_dir, limit):
    """(name, desc (T,5), cent (T,2) local, gt (gx,gy)) using cached uav_descriptors/*.npy
    matched against records_main.csv GT, mirroring phase_a_diag's approach but
    reading the already-computed .npy descriptor cache directly (no re-triangulation)."""
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
    ddir = site_dir / "uav_descriptors"
    pdir = site_dir / "uav_polygons"
    for jf in sorted(pdir.glob("*.json")):
        name = json.loads(jf.read_text(encoding="utf-8")).get("image", jf.stem)
        if name not in gt:
            continue
        npy = ddir / f"{jf.stem}.npy"
        if not npy.exists():
            continue
        desc = np.load(npy)
        if desc.shape[0] == 0:
            continue
        out.append((name, desc, gt[name]))
        if limit and len(out) >= limit:
            break
    return out


def true_match_distances(sat_desc, sat_cent, queries, radius_px, n_samples):
    """For each query frame, for a subset of its triangles, find the L1 distance
    to the geometrically-correct satellite match (nearest centroid to the
    GT-projected position). Returns array of true-match distances + the
    corresponding query descriptor vectors (for reuse in the impostor sweep)."""
    cent_tree = cKDTree(sat_cent)
    dists, qdescs = [], []
    for name, desc, (gx, gy) in queries:
        exp = np.array([gx, gy])  # approx: triangle centroid ~ frame center after GT alignment
        near = cent_tree.query_ball_point(exp, radius_px)
        if not near:
            continue
        near = np.asarray(near)
        for i in range(min(desc.shape[0], 5)):  # cap triangles/frame for speed
            dd = np.abs(sat_desc[near] - desc[i]).sum(axis=1)
            dists.append(float(dd.min()))
            qdescs.append(desc[i])
        if len(dists) >= n_samples:
            break
    return np.asarray(dists), np.asarray(qdescs)


def impostor_curve(sat_desc, qdescs, radii):
    """For each query descriptor, compute the fraction of the FULL satellite
    database within L1 distance r, for each r in `radii`. Returns (N, len(radii))
    array of per-query CDF values, and the overall database size N."""
    N = sat_desc.shape[0]
    cdfs = np.zeros((qdescs.shape[0], len(radii)))
    for i, q in enumerate(qdescs):
        d = np.abs(sat_desc - q).sum(axis=1)
        d.sort()
        # searchsorted for each radius -> count of impostors within r
        counts = np.searchsorted(d, radii, side="right")
        cdfs[i] = counts / N
    return cdfs, N


def pca_effective_dim(desc):
    mu = desc.mean(0)
    X = desc - mu
    cov = (X.T @ X) / X.shape[0]
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    var_explained = eigvals / total if total > 0 else eigvals
    cum = np.cumsum(var_explained)
    # participation ratio: a standard scale-free "effective dimensionality"
    part_ratio = (eigvals.sum() ** 2) / (eigvals ** 2).sum() if (eigvals ** 2).sum() > 0 else float("nan")
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    return {
        "eigenvalues": eigvals.tolist(),
        "variance_explained_pct": [round(100 * v, 2) for v in var_explained],
        "cumulative_pct": [round(100 * c, 2) for c in cum],
        "participation_ratio": round(float(part_ratio), 2),
        "n_components_for_95pct": n95,
    }


def run_site(site, args):
    sd = Path(args.out) / site
    if not (sd / "satellite_descriptors.csv").exists():
        print(f"[skip] {site}: no cached descriptors"); return None
    sat_desc, sat_cent = load_satellite_desc(sd)
    N = sat_desc.shape[0]
    tag = "+zscore" if args.zscore else "baseline"
    print(f"\n===== site {site} [{tag}]: N={N} satellite triangles =====", flush=True)
    mu_z, sd_z = sat_desc.mean(0), sat_desc.std(0)
    sd_z[sd_z < 1e-6] = 1.0
    if args.zscore:
        sat_desc = (sat_desc - mu_z) / sd_z

    pca = pca_effective_dim(sat_desc)
    print(f"[PCA] variance by component (%): {pca['variance_explained_pct']}", flush=True)
    print(f"[PCA] cumulative (%): {pca['cumulative_pct']}", flush=True)
    print(f"[PCA] participation ratio (effective dim): {pca['participation_ratio']}  "
          f"(components for 95% var: {pca['n_components_for_95pct']})", flush=True)

    queries = load_uav_queries(sd, args.limit)
    if not queries:
        print("[skip] no cached UAV query descriptors"); return {"pca": pca}
    if args.zscore:
        queries = [(n, (d - mu_z) / sd_z, gt) for n, d, gt in queries]
    true_d, qdescs = true_match_distances(sat_desc, sat_cent, queries, args.radius, args.n_samples)
    if true_d.size == 0:
        print("[skip] no true-match samples found"); return {"pca": pca}
    med_true = float(np.median(true_d))
    print(f"[true-match] n={true_d.size} median_L1={med_true:.2f} mean={true_d.mean():.2f}", flush=True)

    # impostor curve: sweep radii around and beyond the true-match distance
    radii = np.array(sorted(set(
        list(np.linspace(0.5, med_true * 0.5, 6)) +
        list(np.linspace(med_true * 0.5, med_true * 1.5, 8)) +
        list(np.linspace(med_true * 1.5, med_true * 4, 6))
    )))
    sub = qdescs[: min(args.n_impostor_queries, qdescs.shape[0])]
    cdfs, _ = impostor_curve(sat_desc, sub, radii)
    mean_cdf = cdfs.mean(axis=0)
    expected_impostors = mean_cdf * N

    # the useful, provable statement: E[rank of true match] ~ 1 + N*P(within d_true)
    p_at_true = float(np.interp(med_true, radii, mean_cdf))
    e_rank = 1 + N * p_at_true
    print(f"[impostor] at r=d_true={med_true:.1f}: mean P(impostor within r)={p_at_true:.5f} "
          f"-> E[impostors closer than true match] = {N*p_at_true:.0f}  "
          f"=> E[rank of true match] ~= {e_rank:.0f}", flush=True)
    for K in (5, 20, 100):
        bound = f"{'<< ' if e_rank > 10*K else ''}retrieval@{K} ~ 0" if e_rank > K else f"retrieval@{K} plausible"
        print(f"  vs K={K}: E[rank]={e_rank:.0f}  -> {bound}", flush=True)

    res = {
        "N": N, "pca": pca,
        "true_match": {"n": int(true_d.size), "median_L1": round(med_true, 2), "mean_L1": round(float(true_d.mean()), 2)},
        "impostor_curve": {"radii": radii.tolist(), "mean_cdf": mean_cdf.tolist(),
                           "expected_impostor_count": expected_impostors.tolist()},
        "e_rank_at_true_match_distance": round(e_rank, 1),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01"])
    ap.add_argument("--limit", type=int, default=100, help="UAV query frames to scan")
    ap.add_argument("--radius", type=float, default=30.0, help="px radius for geometrically-correct match")
    ap.add_argument("--n-samples", type=int, default=150, dest="n_samples")
    ap.add_argument("--n-impostor-queries", type=int, default=30, dest="n_impostor_queries")
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--tag", default=None, help="output json suffix, e.g. 'zscore'")
    args = ap.parse_args()
    allr = {}
    for s in args.sites:
        r = run_site(s, args)
        if r:
            allr[s] = r
    FIGDIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    (FIGDIR / f"collision_analysis{suffix}.json").write_text(json.dumps(allr, indent=2))
    print(f"\n[collision] wrote outputs/paper/collision_analysis{suffix}.json", flush=True)


if __name__ == "__main__":
    main()
