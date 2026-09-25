"""Edge-constrained structural graph: lines -> junctions -> CDT (PSLG) -> verification.

Training-free. Used AFTER the global Sim(2) search (never for retrieval --
per-element retrieval hits the discriminability ceiling documented for M0):

  1. structural line filtering: minimum length, merge collinear fragments,
     suppress dense parallel hatching (roof ribs, crop rows);
  2. junctions: intersections of non-parallel segments (L / T / X) + endpoints;
  3. constrained Delaunay triangulation of the planar straight-line graph
     (points + prescribed segments -- CDT's natural input), giving the
     triangle-star of every junction and the free-cone (Ekeland/MFCA) angle
     between constrained edges;
  4. verification of a pose hypothesis T: junction correspondences under T
     (position + incident directions) and topological consistency of
     constrained edges between matched junctions; RANSAC Sim(2) on the
     correspondences gives a continuous refinement.

Coordinates are metres in the frame of the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------
# 1. structural line filtering
# ----------------------------------------------------------------------------


def _angle(seg):
    return math.atan2(seg[3] - seg[1], seg[2] - seg[0]) % math.pi


def filter_lines(segs: np.ndarray, min_len=6.0, merge_ang_deg=5.0, merge_off=1.0, merge_gap=3.0,
                 hatch_radius=3.0, hatch_max=3) -> np.ndarray:
    """segs (N,4) metres -> filtered (M,4)."""
    if len(segs) == 0:
        return segs
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    s = segs[L >= min_len * 0.5].astype(np.float64)
    # --- merge collinear fragments (greedy, by orientation bins) ---
    ang = np.array([_angle(x) for x in s])
    used = np.zeros(len(s), bool); out = []
    mids = 0.5 * (s[:, :2] + s[:, 2:]); tree = cKDTree(mids)
    for i in np.argsort(-np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])):
        if used[i]:
            continue
        used[i] = True
        a = ang[i]; d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]])
        p0 = s[i, :2]
        ts = [float((s[i, :2] - p0) @ d), float((s[i, 2:] - p0) @ d)]
        grew = True
        while grew:
            grew = False
            lo, hi = min(ts), max(ts)
            c = p0 + d * (lo + hi) / 2
            for j in tree.query_ball_point(c, (hi - lo) / 2 + merge_gap + 30):
                if used[j]:
                    continue
                da = abs((ang[j] - a + math.pi / 2) % math.pi - math.pi / 2)
                if da > math.radians(merge_ang_deg):
                    continue
                e = s[j]
                if abs((e[:2] - p0) @ n) > merge_off or abs((e[2:] - p0) @ n) > merge_off:
                    continue
                t1, t2 = sorted([float((e[:2] - p0) @ d), float((e[2:] - p0) @ d)])
                if t1 > hi + merge_gap or t2 < lo - merge_gap:
                    continue
                ts += [t1, t2]; used[j] = True; grew = True
        lo, hi = min(ts), max(ts)
        out.append(np.r_[p0 + d * lo, p0 + d * hi])
    m = np.array(out)
    m = m[np.hypot(m[:, 2] - m[:, 0], m[:, 3] - m[:, 1]) >= min_len]
    if len(m) == 0:
        return m
    # --- suppress dense parallel hatching ---
    ang = np.array([_angle(x) for x in m]); mids = 0.5 * (m[:, :2] + m[:, 2:]); tree = cKDTree(mids)
    keep = np.ones(len(m), bool)
    for i, nb in enumerate(tree.query_ball_point(mids, hatch_radius)):
        par = [j for j in nb if j != i and abs((ang[j] - ang[i] + math.pi / 2) % math.pi - math.pi / 2) < math.radians(8)]
        if len(par) > hatch_max:
            keep[i] = False
    return m[keep]


# ----------------------------------------------------------------------------
# 2. junctions
# ----------------------------------------------------------------------------


@dataclass
class Junction:
    xy: np.ndarray            # (2,)
    kind: str                 # 'L', 'T', 'X', 'E' (free endpoint)
    dirs: np.ndarray          # (k,) incident edge directions in [0, 2pi)


def junctions(segs: np.ndarray, eps=2.0, min_angle_deg=30.0) -> List[Junction]:
    out: List[Junction] = []
    if len(segs) == 0:
        return out
    P, Q = segs[:, :2], segs[:, 2:]
    L = np.hypot(*(Q - P).T)
    mids = 0.5 * (P + Q); tree = cKDTree(mids)
    ang = np.array([_angle(x) for x in segs])
    touched = np.zeros((len(segs), 2), bool)
    seen = set()
    for i in range(len(segs)):
        # bounded radius; a long partner j finds i from its own (larger) query
        for j in tree.query_ball_point(mids[i], L[i] / 2 + 60.0 + eps):
            if j == i or (min(i, j), max(i, j)) in seen:
                continue
            seen.add((min(i, j), max(i, j)))
            da = abs((ang[j] - ang[i] + math.pi / 2) % math.pi - math.pi / 2)
            if da < math.radians(min_angle_deg):
                continue
            d1, d2 = Q[i] - P[i], Q[j] - P[j]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9:
                continue
            w = P[j] - P[i]
            t = (w[0] * d2[1] - w[1] * d2[0]) / den
            u = (w[0] * d1[1] - w[1] * d1[0]) / den
            ti, uj = t * L[i], u * L[j]            # metres along each segment
            if not (-eps <= ti <= L[i] + eps and -eps <= uj <= L[j] + eps):
                continue
            x = P[i] + t * d1
            end_i = min(abs(ti), abs(L[i] - ti)) <= eps
            end_j = min(abs(uj), abs(L[j] - uj)) <= eps
            kind = "L" if (end_i and end_j) else ("T" if (end_i or end_j) else "X")
            dirs = []
            for (p, q, tt, LL, k) in ((P[i], Q[i], ti, L[i], i), (P[j], Q[j], uj, L[j], j)):
                a = math.atan2(q[1] - p[1], q[0] - p[0])
                if tt > eps:
                    dirs.append((a + math.pi) % (2 * math.pi))      # towards P
                if tt < LL - eps:
                    dirs.append(a % (2 * math.pi))                  # towards Q
                if abs(tt) <= eps: touched[k, 0] = True
                if abs(LL - tt) <= eps: touched[k, 1] = True
            out.append(Junction(np.asarray(x), kind, np.sort(np.array(dirs))))
    return out


# ----------------------------------------------------------------------------
# 3. CDT over the planar straight-line graph + triangle-star / free-cone
# ----------------------------------------------------------------------------


@dataclass
class LineGraph:
    segs: np.ndarray
    J: np.ndarray                 # (n,2) junction positions
    kinds: np.ndarray             # (n,) str
    dirs: List[np.ndarray]
    free_cone: np.ndarray         # (n,) largest angular gap between constrained edges (deg)
    adj: dict = field(default_factory=dict)   # junction index -> set of junctions joined by a structural segment
    n_triangles: int = 0


def build_graph(segs: np.ndarray, eps=2.0, cdt: bool = True) -> LineGraph:
    js = junctions(segs, eps)
    if not js:
        return LineGraph(segs, np.zeros((0, 2)), np.array([]), [], np.zeros(0))
    J = np.array([j.xy for j in js]); kinds = np.array([j.kind for j in js]); dirs = [j.dirs for j in js]
    fc = np.array([_free_cone(d) for d in dirs])
    # structural adjacency: consecutive junctions along the same segment
    adj = {i: set() for i in range(len(J))}
    P, Q = segs[:, :2], segs[:, 2:]
    jtree = cKDTree(J)
    for k in range(len(segs)):
        d = Q[k] - P[k]; L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        u = d / L; nrm = np.array([-u[1], u[0]])
        cand = jtree.query_ball_point(0.5 * (P[k] + Q[k]), L / 2 + eps)
        on = [(float((J[i] - P[k]) @ u), i) for i in cand
              if abs(float((J[i] - P[k]) @ nrm)) <= eps and -eps <= float((J[i] - P[k]) @ u) <= L + eps]
        on.sort()
        for (_, a), (_, b) in zip(on, on[1:]):
            adj[a].add(b); adj[b].add(a)
    ntri = 0
    if cdt and len(J) >= 3:
        try:
            import triangle
            edges = np.array([(a, b) for a in adj for b in adj[a] if a < b], np.int32).reshape(-1, 2)
            t = triangle.triangulate({"vertices": J, "segments": edges} if len(edges) else {"vertices": J}, "p" if len(edges) else "")
            ntri = len(t.get("triangles", []))
        except Exception:
            ntri = 0
    return LineGraph(segs, J, kinds, dirs, fc, adj, ntri)


def _free_cone(dirs):
    """Largest angular gap between incident constrained edges (Ekeland free cone, degrees)."""
    if len(dirs) == 0:
        return 360.0
    d = np.sort(np.asarray(dirs) % (2 * math.pi))
    gaps = np.diff(np.r_[d, d[0] + 2 * math.pi])
    return float(np.degrees(gaps.max()))


# ----------------------------------------------------------------------------
# 4. verification of a pose hypothesis + RANSAC Sim(2)
# ----------------------------------------------------------------------------


@dataclass
class Verification:
    n_query: int
    n_matched: int
    match_ratio: float
    topo_consistency: float      # fraction of query structural edges (between matched junctions) present in the map graph
    score: float                 # match_ratio * (0.5 + 0.5 * topo) * (0.5 + 0.5 * ekeland)
    pairs: np.ndarray            # (m,2) indices (query, map)
    ekeland: float = 0.0         # mean free-cone (Ekeland) agreement of matched junctions


def _rot_dirs(dirs, th):
    return (np.asarray(dirs) + th) % (2 * math.pi)


def _dir_match(a, b, tol):
    if len(a) == 0 or len(b) == 0:
        return False
    # every query direction must have a map direction within tol
    diff = np.abs(((a[:, None] - b[None, :]) + math.pi) % (2 * math.pi) - math.pi)
    return bool((diff.min(1) <= tol).all())


def verify(qg: LineGraph, mg: LineGraph, mtree: Optional[cKDTree], theta_rad: float, s: float,
           t_xy: np.ndarray, r=3.0, dir_tol_deg=12.0, use_ekeland: bool = True,
           ekeland_tol_deg: float = 30.0, sigma_e_deg: float = 20.0) -> Verification:
    """Query graph (metres, query frame) vs map graph (metres, map frame) under x -> s R x + t.

    A junction pair is accepted when positions agree (r), incident directions agree
    after rotation and -- with ``use_ekeland`` -- the Ekeland free-cone angles agree
    within ``ekeland_tol_deg``. The free-cone angle is the conference descriptor's
    quantity (largest obstacle-free angular sector at a vertex of the CDT), here
    computed between the constrained (structural) edges of the line-graph CDT; it is
    rotation- and scale-invariant. Its agreement also weights the score.
    """
    n = len(qg.J)
    if n == 0 or mtree is None or len(mg.J) == 0:
        return Verification(n, 0, 0.0, 0.0, 0.0, np.zeros((0, 2), int))
    c, sn = math.cos(theta_rad), math.sin(theta_rad)
    R = np.array([[c, -sn], [sn, c]])
    X = (qg.J @ R.T) * s + t_xy
    d, j = mtree.query(X, k=1)
    tol = math.radians(dir_tol_deg)
    pairs = []
    for i in range(n):
        if d[i] <= r and _dir_match(_rot_dirs(qg.dirs[i], theta_rad), mg.dirs[j[i]], tol):
            if use_ekeland and abs(qg.free_cone[i] - mg.free_cone[j[i]]) > ekeland_tol_deg:
                continue
            pairs.append((i, j[i]))
    pairs = np.array(pairs, int).reshape(-1, 2)
    qmap = dict(pairs.tolist())
    tot = ok = 0
    for a in qmap:
        for b in qg.adj.get(a, ()):
            if b in qmap and a < b:
                tot += 1
                ok += int(qmap[b] in mg.adj.get(qmap[a], ()))
    topo = ok / tot if tot else 0.0
    mr = len(pairs) / n
    ek = float(np.mean(np.exp(-np.abs(qg.free_cone[pairs[:, 0]] - mg.free_cone[pairs[:, 1]]) / sigma_e_deg))) \
        if len(pairs) else 0.0
    sc = mr * (0.5 + 0.5 * topo) * ((0.5 + 0.5 * ek) if use_ekeland else 1.0)
    return Verification(n, len(pairs), mr, topo, sc, pairs, ek)


def ransac_sim2(A: np.ndarray, B: np.ndarray, iters=500, thr=2.0, seed=0):
    """Robust similarity B ~ s R A + t from correspondences (Umeyama on inliers)."""
    if len(A) < 2:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(A), 2, replace=False)
        da, db = A[j] - A[i], B[j] - B[i]
        la, lb = np.linalg.norm(da), np.linalg.norm(db)
        if la < 1e-6:
            continue
        s = lb / la; th = math.atan2(db[1], db[0]) - math.atan2(da[1], da[0])
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        t = B[i] - s * R @ A[i]
        res = np.linalg.norm((A @ R.T) * s + t - B, axis=1)
        inl = res < thr
        if best is None or inl.sum() > best[0].sum():
            best = (inl, s, th, t)
    if best is None or best[0].sum() < 2:
        return None
    inl = best[0]
    s, th, t = _umeyama(A[inl], B[inl])
    return dict(s=s, theta=th, t=t, inliers=int(inl.sum()),
                residual=float(np.median(np.linalg.norm((A[inl] @ _R(th).T) * s + t - B[inl], axis=1))))


def _R(th):
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def _umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / A0.var(0).sum()
    th = math.atan2(R[1, 0], R[0, 0])
    return float(s), float(th), mb - s * R @ ma
