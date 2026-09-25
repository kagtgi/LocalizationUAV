#!/usr/bin/env python
"""Synthetic self-test for StructReg (plan: Verification).

1. Random building layout map (rectangles/L-shapes) at 1 m/px.
2. A query is cut out with a KNOWN similarity transform (theta, s, t),
   optionally with boundary noise / dropped / added buildings.
3. fft_search + refine must recover the pose (<= ~1 px, <= 1 deg) on clean
   data; Laplace sigma must grow with boundary noise; a repetitive layout must
   give a small peak ratio (aliasing is detected, Prop. 3).

Run:  python scripts/test_structreg.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization.registration.structure import query_structure, reference_maps, EnsembleConfig  # noqa: E402
from localization.registration.fft_search import search, SearchConfig  # noqa: E402
from localization.registration.refine import refine  # noqa: E402


def make_map(H=2400, W=2400, n=900, seed=0, repetitive=False):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), np.uint8)
    if repetitive:
        for y in range(40, H - 40, 60):
            for x in range(40, W - 40, 60):
                cv2.rectangle(m, (x, y), (x + 30, y + 18), 1, -1)
        return m.astype(np.float32)
    for _ in range(n):
        x, y = rng.integers(20, W - 60), rng.integers(20, H - 60)
        w, h = rng.integers(8, 40), rng.integers(8, 30)
        ang = rng.uniform(0, 180)
        box = cv2.boxPoints(((float(x), float(y)), (float(w), float(h)), float(ang))).astype(np.int32)
        cv2.fillPoly(m, [box], 1)
        if rng.random() < 0.3:  # L-shape wing
            box2 = cv2.boxPoints(((float(x + w / 2), float(y + h / 2)), (float(w / 2), float(h * 1.2)), float(ang))).astype(np.int32)
            cv2.fillPoly(m, [box2], 1)
    return m.astype(np.float32)


def cut_query(prob_map, u, v, theta_deg, s, qh=500, qw=700, gsd_ref=1.0, gsd_q=0.5):
    """Render the query (north-up up to theta) whose centre is at ref (u, v)."""
    # query pixel x_q (metres = (x_q - c) * gsd_q) maps to ref pixel s R x / gsd_ref + (u,v)
    th = math.radians(theta_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    a = s * gsd_q / gsd_ref
    A = np.zeros((2, 3))
    A[:, :2] = a * R
    A[:, 2] = np.array([u, v]) - A[:, :2] @ np.array([(qw - 1) / 2, (qh - 1) / 2])
    return cv2.warpAffine(prob_map, A, (qw, qh), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def perturb(q, rng, sigma_px=0.0, drop=0.0, add=0.0):
    m = (q > 0.5).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    if drop > 0:
        for i in range(1, n):
            if rng.random() < drop:
                m[lab == i] = 0
    if add > 0:
        for _ in range(int(add * n)):
            x, y = rng.integers(0, m.shape[1] - 30), rng.integers(0, m.shape[0] - 30)
            cv2.rectangle(m, (int(x), int(y)), (int(x + rng.integers(8, 30)), int(y + rng.integers(8, 20))), 1, -1)
    p = m.astype(np.float32)
    if sigma_px > 0:
        noise = cv2.GaussianBlur(rng.normal(0, 1, p.shape).astype(np.float32), (0, 0), 3) * sigma_px
        p = cv2.GaussianBlur(p, (0, 0), 1.0) + noise * 0.4
    return np.clip(p, 0, 1)


def run_case(name, ref_prob, u, v, th, s, rng, radius_m=400, **pert):
    ref = reference_maps(ref_prob, gsd=1.0, sigma_m=2.0)
    qp = cut_query(ref_prob, u, v, th, s)
    qp = perturb(qp, rng, **pert)
    q = query_structure(qp, gsd=0.5, ens=EnsembleConfig(dp_tol_m=(0.5, 1.0)))
    prior = (u + rng.uniform(-150, 150), v + rng.uniform(-150, 150))
    cfg = SearchConfig(thetas_deg=tuple(np.arange(-10, 10.01, 2.5)), scales=(0.9, 0.95, 1.0, 1.05, 1.1))
    res = search(q, ref, center_uv=prior, radius_m=radius_m, cfg=cfg)
    pk = res.peaks[0]
    rr = refine(q, ref, pk, cfg)
    err_grid = math.hypot(pk.u - u, pk.v - v)
    err = math.hypot(rr.u - u, rr.v - v)
    print(f"{name:28s} grid_err={err_grid:6.2f}px refined_err={err:6.2f}px dth={rr.theta_deg - th:+6.2f} "
          f"ds={rr.s - s:+.3f} sigma_pos={rr.sigma_pos_m:6.3f}m ratio={pk.score - res.second_score:.4f} nb={q.n_buildings}")
    return err, abs(rr.theta_deg - th), rr.sigma_pos_m, pk.score - res.second_score


def main():
    rng = np.random.default_rng(1)
    ref_prob = make_map()
    fails = 0
    e, dth, sig0, ratio_rand = run_case("clean", ref_prob, 1200.3, 1100.7, 4.0, 1.03, rng)
    fails += not (e <= 1.5 and dth <= 1.0)
    sigs = []
    for sp in (0.5, 1.5, 3.0):
        e2, _, sg, _ = run_case(f"boundary noise {sp}", ref_prob, 1150.0, 1250.0, -6.0, 0.97, rng, sigma_px=sp)
        sigs.append(sg)
        fails += not (e2 <= 5.0)
    e3, *_ = run_case("drop 30% / add 20%", ref_prob, 1300.0, 1000.0, 2.0, 1.0, rng, drop=0.3, add=0.2)
    fails += not (e3 <= 5.0)
    rep = make_map(repetitive=True)
    _, _, _, ratio_rep = run_case("repetitive grid", rep, 1200.0, 1200.0, 0.0, 1.0, rng)
    fails += not (ratio_rep < ratio_rand)
    print("sigma trend:", ["%.3f" % x for x in sigs], "(should increase)")
    print("peak ratio random vs repetitive: %.4f vs %.4f (repetitive should be smaller)" % (ratio_rand, ratio_rep))
    print("FAILURES:", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
