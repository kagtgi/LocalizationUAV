#!/usr/bin/env python
"""Synthetic test of the oriented line-structure method (known pose).

Map = random building outlines (rotated rectangles, some L-shapes) + a road grid,
as line segments in metres. The query is the SAME structure seen under a known
Sim(2) pose (segments inverse-transformed, so orientation labels rotate
correctly), with segment dropout/jitter. Checks:
  1. oriented search + refine recovers the pose;
  2. oriented peak ratio >= isotropic peak ratio (orientation adds distinctiveness);
  3. line-graph verification (with Ekeland) scores the true pose above a wrong peak.
"""
import math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization.registration import lines as LN
from localization.registration import linegraph as LG
from localization.registration.fft_search import search, SearchConfig
from localization.registration.refine import refine


def rect(cx, cy, w, h, a):
    c, s = math.cos(a), math.sin(a)
    P = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ np.array([[c, s], [-s, c]]) + [cx, cy]
    return [np.r_[P[i], P[(i + 1) % 4]] for i in range(4)]


def make_map(rng, size=1600.0):
    segs = []
    for x in np.arange(100, size, 180):
        segs.append(np.array([x, 0, x + rng.uniform(-20, 20), size]))
    for y in np.arange(60, size, 220):
        segs.append(np.array([0, y, size, y + rng.uniform(-20, 20)]))
    for _ in range(700):
        segs += rect(rng.uniform(30, size - 30), rng.uniform(30, size - 30), rng.uniform(10, 45),
                     rng.uniform(8, 30), rng.uniform(0, math.pi))
    return np.array(segs)


def to_query(segs, u_m, v_m, th, s, half=220.0, rng=None, drop=0.2, jit=0.3):
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = []
    for sg in segs:
        a = (R.T @ (sg[:2] - [u_m, v_m])) / s; b = (R.T @ (sg[2:] - [u_m, v_m])) / s
        if max(abs(a).max(), abs(b).max()) > half * 1.6:
            continue
        if rng.random() < drop:
            continue
        out.append(np.r_[a, b] + rng.normal(0, jit, 4))
    return np.array(out)


def main():
    rng = np.random.default_rng(3)
    segs = make_map(rng)
    gm, gq, K = 0.6, 0.3, 8
    H = W = int(1600 / gm)
    lab = LN.rasterize(segs / gm, (H, W), K=K)
    ref = LN.reference_from_lines(lab, gm, 2.0, K=K)
    u_m, v_m, th, s = 812.3, 777.7, math.radians(6.0), 1.04
    qs = to_query(segs, u_m, v_m, th, s, rng=rng)
    n = int(2 * 220 / gq) | 1; c = (n - 1) / 2
    qlab = LN.rasterize(qs / gq + c, (n, n), K=K)
    q = LN.query_from_lines(qlab, np.ones((n, n), np.float32), gq, np.ones((n, n), np.uint8), K=K)
    prior = (u_m / gm + 120, v_m / gm - 90)
    fails = 0
    res = {}
    for name, ori in (("oriented", True), ("isotropic", False)):
        cfg = SearchConfig(use_mask=False, oriented=ori, thetas_deg=tuple(np.arange(-10, 10.01, 2.5)))
        r = search(q, ref, center_uv=prior, radius_m=400, cfg=cfg)
        rr = refine(q, ref, r.peaks[0], cfg)
        err = math.hypot(rr.u * gm - u_m, rr.v * gm - v_m)
        res[name] = (r, rr, err)
        print(f"{name:10s} err={err:6.2f} m  dth={rr.theta_deg - 6.0:+.2f}  ds={rr.s - s:+.3f}  "
              f"margin={r.peaks[0].score - r.second_score:.4f} rel={(r.peaks[0].score - r.second_score) / r.peaks[0].score:.3f}  sigma={rr.sigma_pos_m:.3f}  K={r.extra['K']}")
    fails += res["oriented"][2] > 2.0
    rel = lambda r: (r.peaks[0].score - r.second_score) / r.peaks[0].score   # scale-free distinctiveness
    print(f"relative margin: oriented {rel(res['oriented'][0]):.3f} vs isotropic {rel(res['isotropic'][0]):.3f}")
    fails += rel(res["oriented"][0]) < rel(res["isotropic"][0])
    # graph verification: true pose vs a shifted (wrong) pose
    mg = LG.build_graph(LG.filter_lines(segs), cdt=False); mt = cKDTree(mg.J)
    qg = LG.build_graph(LG.filter_lines(qs), cdt=True)
    good = LG.verify(qg, mg, mt, th, s, np.array([u_m, v_m]))
    bad = LG.verify(qg, mg, mt, th, s, np.array([u_m + 90, v_m - 60]))
    print(f"graph: query {len(qg.J)} junctions ({qg.n_triangles} CDT tris), map {len(mg.J)}; "
          f"verify true={good.score:.3f} (topo {good.topo_consistency:.2f}, ekeland {good.ekeland:.2f})  wrong={bad.score:.3f}")
    fails += good.score <= bad.score
    rs = LG.ransac_sim2(qg.J[good.pairs[:, 0]], mg.J[good.pairs[:, 1]]) if good.n_matched >= 3 else None
    if rs:
        print(f"ransac: t err {math.hypot(rs['t'][0]-u_m, rs['t'][1]-v_m):.2f} m, dth {math.degrees(rs['theta'])-6:+.2f}, inliers {rs['inliers']}")
    print("FAILURES:", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
