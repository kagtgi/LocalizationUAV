"""Exhaustive pose search on a (theta, s) grid with FFT translation scoring (§4.5).

For a fixed rotation theta and scale s both translation-dependent terms of
J(T) are cross-correlations:

    E(t) = sum_b w_b G(t + q_b)          = (Tpl_w  * G)(t)
    O(t) = sum_u M_U^{th,s}(u) M_S(t+u)  = (Tpl_M * M_S)(t)

so a single FFT per (theta, s) scores *every* translation in the search
window (Proposition 1: the grid optimum is found exactly, not approximately).

Frames: reference raster pixels (col=u east, row=v south) at ``ref.gsd``.
Query points are metres in a north-up frame (x east, y south) centred on the
UAV image centre. A pose maps query x to reference pixel  s R_theta x / gsd + t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

from .structure import QueryStructure, RefMaps


@dataclass
class SearchConfig:
    thetas_deg: Sequence[float] = tuple(np.arange(-10.0, 10.01, 2.5))
    scales: Sequence[float] = (0.9, 0.95, 1.0, 1.05, 1.1)
    alpha: float = 0.5                 # weight of the signed-mask overlap term
    sigma_theta_deg: float = 10.0      # heading prior std (IMU)
    sigma_logs: float = 0.1            # scale prior std (altitude)
    theta0_deg: float = 0.0
    s0: float = 1.0
    use_edge: bool = True
    use_mask: bool = True
    ncc: bool = True                   # footprint-normalized cross-correlation (recommended)
    topk: int = 5
    nms_m: float = 30.0
    second_peak_excl_m: float = 50.0
    batch: int = 8
    max_tile: int = 6144               # FFT tile size for global mode
    device: str = "cuda"


@dataclass
class Peak:
    u: float           # ref pixel col of query centre
    v: float           # ref pixel row of query centre
    theta_deg: float
    s: float
    score: float


@dataclass
class SearchResult:
    peaks: List[Peak]
    second_score: float        # best score > second_peak_excl_m from the top peak
    score_map: Optional[np.ndarray] = None   # max over (theta,s), region coords
    region_origin: tuple = (0, 0)            # (u0, v0) of score_map[0,0]'s query-centre position
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------------


def _pose_points(q: QueryStructure, theta_deg: float, s: float, gsd_ref: float) -> np.ndarray:
    th = math.radians(theta_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
    return (q.pts @ R.T) * (s / gsd_ref)


def _templates(q: QueryStructure, combos, gsd_ref: float, use_mask: bool):
    """Rasterize edge-weight and signed-mask templates for each (theta, s).

    Returns list of (tpl_w, tpl_m, (ox, oy)) where (ox, oy) is the template
    pixel position of the query centre.
    """
    hq, wq = q.mask.shape
    cxq, cyq = (wq - 1) / 2.0, (hq - 1) / 2.0
    corners = np.array([[-cxq, -cyq], [cxq, -cyq], [cxq, cyq], [-cxq, cyq]], np.float32) * q.gsd
    out = []
    for th, s in combos:
        thr = math.radians(th)
        R = np.array([[math.cos(thr), -math.sin(thr)], [math.sin(thr), math.cos(thr)]], np.float32)
        k = s / gsd_ref
        cc = (corners @ R.T) * k
        mn = np.floor(cc.min(0)) - 2
        mx = np.ceil(cc.max(0)) + 2
        W, H = int(mx[0] - mn[0]) + 1, int(mx[1] - mn[1]) + 1
        ox, oy = -mn[0], -mn[1]
        # edge template: bilinear splat of weighted points
        p = (q.pts @ R.T) * k + np.array([ox, oy], np.float32)
        tw = np.zeros((H, W), np.float32)
        x0 = np.floor(p[:, 0]).astype(int); y0 = np.floor(p[:, 1]).astype(int)
        fx = p[:, 0] - x0; fy = p[:, 1] - y0
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = np.clip(x0 + dx, 0, W - 1), np.clip(y0 + dy, 0, H - 1)
            np.add.at(tw, (yi, xi), q.w * ww)
        # affine: query pixel -> template pixel
        a = k * q.gsd
        A = np.array([[a * R[0, 0], a * R[0, 1], 0], [a * R[1, 0], a * R[1, 1], 0]], np.float32)
        A[:, 2] = np.array([ox, oy]) - A[:, :2] @ np.array([cxq, cyq], np.float32)
        tm = cv2.warpAffine(q.mask, A, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.0) if use_mask else None
        tv = cv2.warpAffine(q.valid.astype(np.float32), A, (W, H), flags=cv2.INTER_NEAREST, borderValue=0.0)
        out.append((tw, tm, (ox, oy), tv))
    return out


# ----------------------------------------------------------------------------
# correlation
# ----------------------------------------------------------------------------


def _xcorr_batch(region: torch.Tensor, tpls: List[np.ndarray], device) -> torch.Tensor:
    """Valid cross-correlation of one region with a batch of templates.

    Returns (B, H-hmax+1, W-wmax+1) where result[b, t] = sum_u tpl_b[u] region[t+u].
    Templates are zero-padded at bottom/right to a common (hmax, wmax).
    """
    H, W = region.shape
    hmax = max(t.shape[0] for t in tpls); wmax = max(t.shape[1] for t in tpls)
    T = torch.zeros((len(tpls), H, W), device=device, dtype=torch.float32)
    for i, t in enumerate(tpls):
        T[i, : t.shape[0], : t.shape[1]] = torch.from_numpy(t).to(device)
    Fr = torch.fft.rfft2(region)
    Ft = torch.fft.rfft2(T)
    c = torch.fft.irfft2(torch.conj(Ft) * Fr[None], s=(H, W))
    return c[:, : H - hmax + 1, : W - wmax + 1]


def _ncc(region, tpls, V, N, device, cache, key, eps):
    """Batched normalized cross-correlation over each template's footprint V_b.

    ncc_b(t) = [sum_u T_b(u) S(t+u) - (sum T_b) mu_b(t)] / (N_b sd(T_b) sd_b(t)),
    mu_b(t), sd_b(t): mean / std of S under the footprint V_b placed at t.
    Everything is a cross-correlation, hence exact by FFT (Prop. 1 still holds).
    """
    H, W = region.shape
    hmax = max(t.shape[0] for t in tpls); wmax = max(t.shape[1] for t in tpls)
    if key not in cache:
        cache[key] = (torch.fft.rfft2(region), torch.fft.rfft2(region * region))
    F1, F2 = cache[key]
    def pad(arrs):
        T = torch.zeros((len(arrs), H, W), device=device, dtype=torch.float32)
        for i, a in enumerate(arrs):
            T[i, : a.shape[0], : a.shape[1]] = torch.from_numpy(a).to(device)
        return T
    Tt = pad([t * v for t, v in zip(tpls, V)]); Vt = pad(V)
    FT = torch.conj(torch.fft.rfft2(Tt)); FV = torch.conj(torch.fft.rfft2(Vt))
    sl = (slice(None), slice(0, H - hmax + 1), slice(0, W - wmax + 1))
    A = torch.fft.irfft2(FT * F1[None], s=(H, W))[sl]
    B = torch.fft.irfft2(FV * F1[None], s=(H, W))[sl]
    C = torch.fft.irfft2(FV * F2[None], s=(H, W))[sl]
    mu = B / N
    var = (C / N - mu * mu).clamp_min(0)
    sT = Tt.sum((1, 2))[:, None, None]
    mT = sT / N
    vT = ((Tt * Tt).sum((1, 2))[:, None, None] / N - mT * mT).clamp_min(1e-12)
    return (A - sT * mu) / (N * torch.sqrt(vT) * torch.sqrt(var + eps * eps))


def search(
    q: QueryStructure,
    ref: RefMaps,
    center_uv: Optional[tuple] = None,
    radius_m: Optional[float] = None,
    cfg: Optional[SearchConfig] = None,
    keep_map: bool = False,
) -> SearchResult:
    """Grid-exhaustive maximization of J over (theta, s, t).

    ``center_uv``/``radius_m`` define the prior window (reference pixels / metres)
    for the query centre; ``None`` searches the whole reference map (global mode).
    """
    cfg = cfg or SearchConfig()
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    combos = [(float(th), float(s)) for th in cfg.thetas_deg for s in cfg.scales]
    tpls = _templates(q, combos, ref.gsd, cfg.use_mask)
    hmax = max(t[0].shape[0] for t in tpls); wmax = max(t[0].shape[1] for t in tpls)
    Hs, Ws = ref.G.shape

    # search window for the query centre -> region of the reference map
    if center_uv is None:
        cu0, cu1, cv0, cv1 = 0, Ws, 0, Hs
    else:
        r = radius_m / ref.gsd
        cu0, cu1 = int(center_uv[0] - r), int(center_uv[0] + r) + 1
        cv0, cv1 = int(center_uv[1] - r), int(center_uv[1] + r) + 1

    # prior penalties per combo
    pri = np.array([
        -((th - cfg.theta0_deg) ** 2) / (2 * cfg.sigma_theta_deg ** 2)
        - (math.log(s) - math.log(cfg.s0)) ** 2 / (2 * cfg.sigma_logs ** 2)
        for th, s in combos
    ], np.float32) * 0.01  # priors are soft tie-breakers at the J scale (J in [~-1, ~2])

    wsum = float(q.w.sum()) + 1e-6

    # tile the query-centre window so each FFT region stays <= max_tile
    step = max(256, cfg.max_tile - max(hmax, wmax))
    best = None  # (score_map, arg_map, u0, v0) pieces
    pieces = []
    for tv in range(cv0, cv1, step):
        for tu in range(cu0, cu1, step):
            wv1, wu1 = min(tv + step, cv1), min(tu + step, cu1)
            # region: query centre at (u,v) needs ref pixels [u-ox, u-ox+wmax)
            # we use a common origin: region covers [tu - OXmax, wu1 - OXmin + wmax]
            ox_all = np.array([t[2][0] for t in tpls]); oy_all = np.array([t[2][1] for t in tpls])
            ru0 = int(math.floor(tu - ox_all.max())); rv0 = int(math.floor(tv - oy_all.max()))
            ru1 = int(math.ceil(wu1 - ox_all.min())) + wmax; rv1 = int(math.ceil(wv1 - oy_all.min())) + hmax
            Hr, Wr = rv1 - rv0, ru1 - ru0
            Gr = np.zeros((Hr, Wr), np.float32); Mr = np.zeros((Hr, Wr), np.float32)
            a0, a1 = max(rv0, 0), min(rv1, Hs); b0, b1 = max(ru0, 0), min(ru1, Ws)
            if a1 <= a0 or b1 <= b0:
                continue
            Gr[a0 - rv0 : a1 - rv0, b0 - ru0 : b1 - ru0] = ref.G[a0:a1, b0:b1]
            Mr[a0 - rv0 : a1 - rv0, b0 - ru0 : b1 - ru0] = ref.M[a0:a1, b0:b1]
            Gt = torch.from_numpy(Gr).to(dev); Mt = torch.from_numpy(Mr).to(dev)
            nu, nv = wu1 - tu, wv1 - tv
            smax = torch.full((nv, nu), -1e9, device=dev); sarg = torch.zeros((nv, nu), dtype=torch.long, device=dev)
            Fcache = {}
            for b0i in range(0, len(combos), cfg.batch):
                idx = list(range(b0i, min(b0i + cfg.batch, len(combos))))
                tot = None
                if cfg.ncc:
                    # footprint-normalized cross-correlation (zero-mean, unit-variance
                    # over the camera footprint V): removes the bias of raw scores
                    # towards dense-building / empty areas (Lewis-style NCC by FFT).
                    V = [tpls[i][3] for i in idx]
                    N = torch.tensor([float(v.sum()) + 1e-6 for v in V], device=dev)[:, None, None]
                    if cfg.use_edge:
                        tot = _ncc(Gt, [tpls[i][0] for i in idx], V, N, dev, Fcache, "G", eps=0.05)
                    if cfg.use_mask:
                        cm = _ncc(Mt, [tpls[i][1] for i in idx], V, N, dev, Fcache, "M", eps=0.1)
                        tot = cfg.alpha * cm if tot is None else tot + cfg.alpha * cm
                else:
                    if cfg.use_edge:
                        tot = _xcorr_batch(Gt, [tpls[i][0] for i in idx], dev) / wsum
                    if cfg.use_mask:
                        msum = [float(np.abs(tpls[i][1]).sum()) + 1e-6 for i in idx]
                        cm = _xcorr_batch(Mt, [tpls[i][1] for i in idx], dev)
                        cm = cm / torch.tensor(msum, device=dev)[:, None, None]
                        tot = cfg.alpha * cm if tot is None else tot + cfg.alpha * cm
                for j, i in enumerate(idx):
                    ox, oy = tpls[i][2]
                    # query centre (u,v) <-> template origin t = (u - ox, v - oy) - (ru0, rv0)
                    su = int(round(tu - ox)) - ru0; sv = int(round(tv - oy)) - rv0
                    sl = tot[j, sv : sv + nv, su : su + nu] + float(pri[i])
                    upd = sl > smax[: sl.shape[0], : sl.shape[1]]
                    smax[: sl.shape[0], : sl.shape[1]] = torch.where(upd, sl, smax[: sl.shape[0], : sl.shape[1]])
                    sarg[: sl.shape[0], : sl.shape[1]] = torch.where(upd, torch.full_like(sarg[: sl.shape[0], : sl.shape[1]], i), sarg[: sl.shape[0], : sl.shape[1]])
            pieces.append((smax.cpu().numpy(), sarg.cpu().numpy(), tu, tv))

    # stitch pieces into one window map
    U0, V0 = cu0, cv0
    Smap = np.full((cv1 - cv0, cu1 - cu0), -1e9, np.float32)
    Amap = np.zeros((cv1 - cv0, cu1 - cu0), np.int64)
    for sm, sa, tu, tv in pieces:
        Smap[tv - V0 : tv - V0 + sm.shape[0], tu - U0 : tu - U0 + sm.shape[1]] = sm
        Amap[tv - V0 : tv - V0 + sa.shape[0], tu - U0 : tu - U0 + sa.shape[1]] = sa

    # restrict to circular window in prior mode
    if center_uv is not None:
        yy, xx = np.mgrid[V0:cv1, U0:cu1]
        out = (xx - center_uv[0]) ** 2 + (yy - center_uv[1]) ** 2 > (radius_m / ref.gsd) ** 2
        Smap[out] = -1e9

    peaks = _nms_peaks(Smap, Amap, combos, U0, V0, cfg.topk, cfg.nms_m / ref.gsd)
    second = -1e9
    if peaks:
        yy, xx = np.mgrid[V0:cv1, U0:cu1]
        far = (xx - peaks[0].u) ** 2 + (yy - peaks[0].v) ** 2 > (cfg.second_peak_excl_m / ref.gsd) ** 2
        if far.any():
            second = float(Smap[far].max())
    return SearchResult(peaks=peaks, second_score=second,
                        score_map=Smap if keep_map else None, region_origin=(U0, V0))


def _nms_peaks(S, A, combos, U0, V0, k, r_px):
    S = S.copy()
    out = []
    for _ in range(k):
        i = int(np.argmax(S))
        v, u = divmod(i, S.shape[1])
        sc = float(S[v, u])
        if sc <= -1e8:
            break
        th, s = combos[int(A[v, u])]
        out.append(Peak(u=float(u + U0), v=float(v + V0), theta_deg=th, s=s, score=sc))
        r = int(math.ceil(r_px))
        S[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1] = -1e9
    return out
