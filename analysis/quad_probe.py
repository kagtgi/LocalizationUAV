#!/usr/bin/env python
"""Astrometry.net-style quad geometric hashing for UAV<->satellite building
constellation matching. THE GATE for the "full upside" narrative: does
similarity-invariant 4-point hashing over building centroids escape the
per-element / per-cell discriminability ceiling measured elsewhere this
session (0% retrieval, ~0-10% area recall, km-scale error)?

Pipeline per site:
  1. Satellite: for every building centroid, form a quad = {building, its 3
     nearest neighbours}. Compute a similarity-invariant 4-D hash code
     (Lang et al. 2010 "astrometry.net" convention: two farthest points -> A,B
     mapped to (0,0),(1,1); the other two -> C,D expressed in that frame).
     Index all codes in a cKDTree. Cached to quad_cache.npz (like ecd_sat_cache).
  2. UAV frame: same quad construction over the frame's local building
     centroids. For each frame quad, look up top-K nearest satellite quad
     codes. Each candidate gives a 4-point correspondence -> closed-form
     similarity transform (Umeyama). Apply to ALL of the frame's centroids;
     count satellite-centroid inliers within a radius. Keep the
     highest-inlier transform per frame; predicted position = transform
     applied to the frame's nadir centre (CENTER,CENTER).
  3. Ekeland-prune ablation: re-score candidates by adding a penalty for
     per-building Ekeland-shape mismatch between corresponding points, so the
     angle's marginal contribution is measured directly (hash-only vs
     hash+Ekeland-prune).

Self-test (--selftest, no VM data): synthetic 4-point sets under a random
similarity transform must produce (near-)identical codes.
"""
from __future__ import annotations
import argparse, json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ecd_probe import load_sat_buildings, load_uav_buildings, CENTER
from localization.io.bounds import pixel_offset_to_meters

FIGDIR = Path("outputs/paper")


# --------------------------------------------------------------------------- quad code

def quad_code(pts):
    """pts: (4,2). Returns (code (4,), order (ai,bi,ci,di) indices into pts) or (None, None)."""
    best = None
    for i, j in combinations(range(4), 2):
        d = float(np.hypot(*(pts[i] - pts[j])))
        if best is None or d > best[0]:
            best = (d, i, j)
    _, ai, bi = best
    ci, di = [k for k in range(4) if k not in (ai, bi)]

    def _xform(a, b, c, d):
        A, B, C, D = pts[a], pts[b], pts[c], pts[d]
        ab = B - A
        L = float(np.hypot(*ab))
        if L < 1e-9:
            return None
        theta = float(np.arctan2(ab[1], ab[0]))
        rot = np.pi / 4 - theta
        scale = np.sqrt(2.0) / L
        cr, sr = np.cos(rot), np.sin(rot)
        R = np.array([[cr, -sr], [sr, cr]])
        Cc = (R @ (C - A)) * scale
        Dc = (R @ (D - A)) * scale
        return Cc, Dc

    res = _xform(ai, bi, ci, di)
    if res is None:
        return None, None
    Cc, Dc = res
    if (Cc[0], Cc[1]) > (Dc[0], Dc[1]):
        Cc, Dc = Dc, Cc
        ci, di = di, ci
    if Cc[0] + Dc[0] > 1.0:
        Cc, Dc = 1.0 - Cc, 1.0 - Dc
        ai, bi = bi, ai
        if (Cc[0], Cc[1]) > (Dc[0], Dc[1]):
            Cc, Dc = Dc, Cc
            ci, di = di, ci
    code = np.array([Cc[0], Cc[1], Dc[0], Dc[1]], np.float64)
    return code, (ai, bi, ci, di)


def selftest():
    rng = np.random.default_rng(0)
    ok = 0
    for trial in range(200):
        pts = rng.uniform(-5, 5, size=(4, 2))
        if np.linalg.matrix_rank(pts - pts.mean(0)) < 2:
            continue
        code0, order0 = quad_code(pts)
        if code0 is None:
            continue
        # random similarity transform: rotation, scale, translation
        theta = rng.uniform(0, 2 * np.pi)
        s = rng.uniform(0.3, 3.0)
        t = rng.uniform(-10, 10, size=2)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts2 = (s * (R @ pts.T).T) + t
        code1, order1 = quad_code(pts2)
        if code1 is None:
            continue
        if np.allclose(code0, code1, atol=1e-6):
            ok += 1
        else:
            print(f"  [FAIL] trial {trial}: code0={code0} code1={code1}")
    print(f"[selftest] quad_code invariant under similarity transform: {ok}/200 passed", flush=True)

    # umeyama self-test: recover the exact transform from 4 known correspondences
    rng = np.random.default_rng(1)
    ok2 = 0
    for trial in range(200):
        src = rng.uniform(-5, 5, size=(4, 2))
        theta = rng.uniform(0, 2 * np.pi); s = rng.uniform(0.3, 3.0); t = rng.uniform(-10, 10, size=2)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        tgt = (s * (R @ src.T).T) + t
        s_hat, R_hat, t_hat = umeyama(src, tgt)
        pred = s_hat * (R_hat @ src.T).T + t_hat
        if np.allclose(pred, tgt, atol=1e-6):
            ok2 += 1
    print(f"[selftest] umeyama exact recovery on noiseless 4-pt correspondence: {ok2}/200 passed", flush=True)


def umeyama(src, tgt):
    """Closed-form similarity transform (scale, rotation, translation): tgt ~= s*R@src + t."""
    mu_s = src.mean(0); mu_t = tgt.mean(0)
    sc = src - mu_s; tc = tgt - mu_t
    n = src.shape[0]
    Sigma = (tc.T @ sc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (sc ** 2).sum(axis=1).mean()
    scale = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    t = mu_t - scale * (R @ mu_s)
    return scale, R, t


# --------------------------------------------------------------------------- quad construction

def build_quads(cents, cache_path=None):
    """For every point, quad = {point, 3 nearest neighbours}. Returns (codes (M,4),
    quad_pts (M,4,2) in canonical A,B,C,D order, anchor_idx (M,) original point index)."""
    if cache_path is not None and cache_path.exists():
        z = np.load(cache_path)
        return z["codes"], z["quad_pts"], z["anchor_idx"]
    n = cents.shape[0]
    if n < 4:
        return np.zeros((0, 4)), np.zeros((0, 4, 2)), np.zeros((0,), int)
    tree = cKDTree(cents)
    _, idx = tree.query(cents, k=4)
    idx = np.atleast_2d(idx)
    codes, quad_pts, anchors = [], [], []
    for i in range(n):
        nn = [j for j in idx[i] if j != i][:3]
        if len(nn) < 3:
            continue
        pts = cents[[i, nn[0], nn[1], nn[2]]]
        code, order = quad_code(pts)
        if code is None:
            continue
        codes.append(code)
        quad_pts.append(pts[list(order)])
        anchors.append(i)
    codes = np.vstack(codes) if codes else np.zeros((0, 4))
    quad_pts = np.stack(quad_pts) if quad_pts else np.zeros((0, 4, 2))
    anchors = np.asarray(anchors, int)
    if cache_path is not None:
        np.savez(cache_path, codes=codes, quad_pts=quad_pts, anchor_idx=anchors)
    return codes, quad_pts, anchors


# --------------------------------------------------------------------------- matching

def match_frame(uav_cents, sat_codes, sat_quad_pts, sat_tree, sat_cent_tree,
                 topk, inlier_radius_px, scale_lo, scale_hi, rot_max_deg=180.0,
                 uav_shapes=None, sat_shape_by_anchor=None, sat_anchor_idx=None,
                 ekeland_lambda=0.0):
    """Returns (best_inliers, best_transform_or_None, n_candidates_tried)."""
    u_codes, u_quad_pts, u_anchor = build_quads(uav_cents)
    if u_codes.shape[0] == 0:
        return 0, None, 0
    best_inl = 0
    best_xform = None
    n_tried = 0
    rot_max = np.radians(rot_max_deg)
    for qi in range(u_codes.shape[0]):
        code = u_codes[qi]
        k = min(topk, sat_codes.shape[0])
        if k == 0:
            continue
        dist, cand = sat_tree.query(code, k=k)
        cand = np.atleast_1d(cand)
        for cj in cand:
            n_tried += 1
            src = u_quad_pts[qi]           # (4,2) uav-local
            tgt = sat_quad_pts[cj]         # (4,2) sat-global
            s_hat, R_hat, t_hat = umeyama(src, tgt)
            if not (scale_lo < s_hat < scale_hi):    # GSD is calibrated (~1.0); reject degenerate scale
                continue
            rot_angle = abs(np.arctan2(R_hat[1, 0], R_hat[0, 0]))
            rot_angle = min(rot_angle, 2 * np.pi - rot_angle)
            if rot_angle > rot_max:   # frames are yaw-aligned in preprocessing; true rotation ~0
                continue
            proj = s_hat * (R_hat @ uav_cents.T).T + t_hat
            # inlier count: projected uav buildings with a nearby sat building
            cnts = sat_cent_tree.query_ball_point(proj, inlier_radius_px, return_length=True)
            inl = int(np.sum(np.asarray(cnts) > 0))
            if ekeland_lambda > 0 and uav_shapes is not None and sat_shape_by_anchor is not None:
                # penalize mismatch on the anchor building's Ekeland shape (min_e, mean_e)
                ua = u_anchor[qi]
                sa = sat_anchor_idx[cj]
                mism = float(np.abs(uav_shapes[ua][:2] - sat_shape_by_anchor[sa][:2]).sum())
                inl = inl - ekeland_lambda * mism
            if inl > best_inl:
                best_inl = inl
                best_xform = (s_hat, R_hat, t_hat)
    return best_inl, best_xform, n_tried


def null_check(site, args):
    """Diagnostic: is the inlier-counting verification step itself informative,
    independent of quad-hash retrieval? Compare inlier count under the KNOWN
    ground-truth transform vs under random wrong translations (same scale/rot)."""
    sd = Path(args.out) / site
    feats, cents = load_sat_buildings(sd, args.max_depth, args.patch_size, args.stride, args.dedup_cell)
    if not feats:
        print(f"[skip] {site}"); return
    sat_cent_tree = cKDTree(cents)
    frames = load_uav_buildings(sd, args.max_depth, args.limit)
    rng = np.random.default_rng(0)
    gt_inl, null_inl = [], []
    xr = (float(cents[:, 0].min()), float(cents[:, 0].max()))
    yr = (float(cents[:, 1].min()), float(cents[:, 1].max()))
    scale = 0.957  # session-calibrated median UAV/satellite pixel-scale ratio
    for name, ufeat, ucent, (gx, gy) in frames:
        if ucent.shape[0] == 0:
            continue
        offset = ucent - np.array([CENTER, CENTER])
        # GT transform: identity rotation, calibrated scale, translation = GT
        proj_gt = scale * offset + np.array([gx, gy])
        cnts = sat_cent_tree.query_ball_point(proj_gt, args.inlier_radius, return_length=True)
        gt_inl.append(int(np.sum(np.asarray(cnts) > 0)))
        # null: same scale/rotation, random translation elsewhere on the map
        rx = rng.uniform(*xr); ry = rng.uniform(*yr)
        proj_null = scale * offset + np.array([rx, ry])
        cnts2 = sat_cent_tree.query_ball_point(proj_null, args.inlier_radius, return_length=True)
        null_inl.append(int(np.sum(np.asarray(cnts2) > 0)))
    gt_inl = np.asarray(gt_inl); null_inl = np.asarray(null_inl)
    bldg_per_frame = np.median([uc.shape[0] for _, _, uc, _ in frames]) if frames else 0
    print(f"\n===== NULL CHECK site {site} (n={len(gt_inl)} frames, radius={args.inlier_radius}px) =====")
    print(f"  GT-transform inliers:     median={np.median(gt_inl):.1f} mean={gt_inl.mean():.1f} "
          f"(median buildings/frame ~{bldg_per_frame:.0f})")
    print(f"  random-translation inliers: median={np.median(null_inl):.1f} mean={null_inl.mean():.1f}")
    print(f"  SNR (GT median / null median): {np.median(gt_inl)/max(np.median(null_inl),0.1):.2f}x", flush=True)


def run_site(site, args):
    sd = Path(args.out) / site
    feats, cents = load_sat_buildings(sd, args.max_depth, args.patch_size, args.stride, args.dedup_cell)
    if not feats:
        print(f"[skip] {site}: no cached satellite buildings"); return None
    shapes = np.vstack([f["shape"] for f in feats])
    meta = json.loads((sd / "site_meta.json").read_text())
    bounds, sw, sh_ = meta["bounds"], meta["sat_width"], meta["sat_height"]

    cache = sd / "quad_cache.npz"
    sat_codes, sat_quad_pts, sat_anchor_idx = build_quads(cents, cache_path=cache)
    sat_tree = cKDTree(sat_codes) if sat_codes.shape[0] else None
    sat_cent_tree = cKDTree(cents)
    print(f"\n===== site {site}: {cents.shape[0]} sat buildings -> {sat_codes.shape[0]} quads =====",
          flush=True)
    if sat_tree is None:
        print("[skip] no satellite quads"); return None

    frames = load_uav_buildings(sd, args.max_depth, args.limit)
    results = {}
    for use_ekeland in ([False, True] if args.ekeland_ablation else [False]):
        errs, inliers_list, scales_list, gt_hits = [], [], [], 0
        n_frames_matched = 0
        for name, ufeat, ucent, (gx, gy) in frames:
            if ucent.shape[0] < 4:
                continue
            ushapes = np.vstack([f["shape"] for f in ufeat])
            best_inl, xform, n_tried = match_frame(
                ucent, sat_codes, sat_quad_pts, sat_tree, sat_cent_tree,
                args.topk, args.inlier_radius, args.scale_lo, args.scale_hi, args.rot_max_deg,
                uav_shapes=ushapes, sat_shape_by_anchor=shapes, sat_anchor_idx=sat_anchor_idx,
                ekeland_lambda=(args.ekeland_lambda if use_ekeland else 0.0))
            inliers_list.append(best_inl)
            if xform is None:
                continue
            n_frames_matched += 1
            s_hat, R_hat, t_hat = xform
            scales_list.append(s_hat)
            center = np.array([CENTER, CENTER])
            pred = s_hat * (R_hat @ center) + t_hat
            err = pixel_offset_to_meters(pred[0] - gx, pred[1] - gy, bounds, sw, sh_)["distance_m"]
            errs.append(err)
            if best_inl >= args.good_inliers:
                gt_hits += 1
        errs = np.asarray(errs)
        inliers_arr = np.asarray(inliers_list)
        scales_arr = np.asarray(scales_list)
        mode = "ekeland" if use_ekeland else "hash_only"
        res = {
            "n_frames": len(frames), "n_matched": n_frames_matched,
            "mutual_quad_coverage_pct": round(100.0 * np.mean(inliers_arr >= args.good_inliers), 1) if len(inliers_arr) else None,
            "median_inliers": float(np.median(inliers_arr)) if len(inliers_arr) else None,
            "median_winning_scale": round(float(np.median(scales_arr)), 3) if len(scales_arr) else None,
            "median_err_m": round(float(np.median(errs)), 1) if len(errs) else None,
            "mean_err_m": round(float(np.mean(errs)), 1) if len(errs) else None,
            "pct_lt_30m": round(100.0 * np.mean(errs < 30), 1) if len(errs) else None,
            "pct_lt_50m": round(100.0 * np.mean(errs < 50), 1) if len(errs) else None,
            "pct_lt_100m": round(100.0 * np.mean(errs < 100), 1) if len(errs) else None,
            "pct_lt_500m": round(100.0 * np.mean(errs < 500), 1) if len(errs) else None,
        }
        results[mode] = res
        print(f"[{mode:9s}] matched={n_frames_matched}/{len(frames)}  "
              f"mutual_quad_cov={res['mutual_quad_coverage_pct']}%  median_inliers={res['median_inliers']}  "
              f"median_scale={res['median_winning_scale']}  "
              f"median_err={res['median_err_m']}m  <30m={res['pct_lt_30m']}% <100m={res['pct_lt_100m']}% <500m={res['pct_lt_500m']}%",
              flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--sites", nargs="+", default=["01"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--topk", type=int, default=15, help="nearest satellite quads to verify per UAV quad")
    ap.add_argument("--inlier-radius", type=float, default=25.0, dest="inlier_radius")
    ap.add_argument("--good-inliers", type=int, default=4, dest="good_inliers")
    ap.add_argument("--scale-lo", type=float, default=0.1, dest="scale_lo")
    ap.add_argument("--scale-hi", type=float, default=10.0, dest="scale_hi")
    ap.add_argument("--rot-max-deg", type=float, default=180.0, dest="rot_max_deg")
    ap.add_argument("--null-check", action="store_true", dest="null_check",
                     help="diagnostic: compare GT-transform inlier count vs random-transform null")
    ap.add_argument("--max-depth", type=int, default=4, dest="max_depth")
    ap.add_argument("--patch-size", type=int, default=500, dest="patch_size")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--dedup-cell", type=float, default=250.0, dest="dedup_cell")
    ap.add_argument("--ekeland-ablation", action="store_true", dest="ekeland_ablation")
    ap.add_argument("--ekeland-lambda", type=float, default=0.02, dest="ekeland_lambda")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.null_check:
        for s in args.sites:
            null_check(s, args)
        return

    allr = {}
    for s in args.sites:
        r = run_site(s, args)
        if r:
            allr[s] = r
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "quad_probe.json").write_text(json.dumps(allr, indent=2))
    print("\n[quad] wrote outputs/paper/quad_probe.json", flush=True)


if __name__ == "__main__":
    main()
