"""Exhaustive pose search on a (theta, s) grid with FFT translation scoring (§4.5).

For a fixed rotation theta and scale s, every translation-dependent statistic
of the objective is a cross-correlation, so one batch of FFTs per (theta, s)
scores *every* translation of the search window exactly (Proposition 1).

Edge evidence may be isotropic (one kernel map G) or ORIENTED: K orientation
channels G_k built from structure whose direction falls in bin k. A query
boundary sample of orientation phi, rotated by theta, votes in channel
bin(phi + theta). The score is a masked normalized cross-correlation over the
stacked (channel x footprint) domain:

    A(t)   = sum_k (T_k * G_k)(t)                 (one inverse FFT after summing spectra)
    ncc(t) = [A - S_T S_G / N] / sqrt(var_T * (var_G + N eps^2))

with S_G, var_G taken over the overlap of the footprint with the valid map, so
the score is invariant to structure density (no bias to busy or empty areas).
Oriented matching is the directional-chamfer idea (edges must also agree in
direction), obtained here exactly and globally over Sim(2).

Frames: reference raster pixels (col=u east, row=v south) at ``ref.gsd``.
Query points are metres (x east, y south) centred on the query image centre.
A pose maps query x to reference pixel  s R_theta x / gsd + t.
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
    alpha: float = 0.5                 # weight of the signed-mask (region) term
    sigma_theta_deg: float = 10.0      # heading prior std (IMU)
    sigma_logs: float = 0.1            # scale prior std (altitude)
    theta0_deg: float = 0.0
    s0: float = 1.0
    use_edge: bool = True
    use_mask: bool = True
    oriented: bool = True              # use orientation channels when both sides provide them
    ncc: bool = True                   # footprint-normalized (masked) cross-correlation
    zscore: bool = True                # standardize each (theta, s) score map (removes template-size bias)
    topk: int = 5
    nms_m: float = 30.0
    second_peak_excl_m: float = 50.0
    batch: int = 2
    max_tile: int = 6144               # FFT tile size for global mode
    device: str = "cuda"


@dataclass
class Peak:
    u: float
    v: float
    theta_deg: float
    s: float
    score: float


@dataclass
class SearchResult:
    peaks: List[Peak]
    second_score: float
    score_map: Optional[np.ndarray] = None
    region_origin: tuple = (0, 0)
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# template geometry + lazy rasterization
# ----------------------------------------------------------------------------


def _geom(q: QueryStructure, th: float, s: float, gsd_ref: float):
    hq, wq = q.valid.shape
    cxq, cyq = (wq - 1) / 2.0, (hq - 1) / 2.0
    thr = math.radians(th)
    R = np.array([[math.cos(thr), -math.sin(thr)], [math.sin(thr), math.cos(thr)]], np.float32)
    k = s / gsd_ref
    corners = np.array([[-cxq, -cyq], [cxq, -cyq], [cxq, cyq], [-cxq, cyq]], np.float32) * q.gsd
    cc = (corners @ R.T) * k
    mn = np.floor(cc.min(0)) - 2
    mx = np.ceil(cc.max(0)) + 2
    W, H = int(mx[0] - mn[0]) + 1, int(mx[1] - mn[1]) + 1
    return R, k, W, H, float(-mn[0]), float(-mn[1]), cxq, cyq


def _splat(p, w, H, W):
    out = np.zeros((H, W), np.float32)
    if len(p) == 0:
        return out
    x0 = np.floor(p[:, 0]).astype(int); y0 = np.floor(p[:, 1]).astype(int)
    fx = p[:, 0] - x0; fy = p[:, 1] - y0
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = np.clip(x0 + dx, 0, W - 1), np.clip(y0 + dy, 0, H - 1)
        np.add.at(out, (yi, xi), w * ww)
    return out


def _template(q: QueryStructure, th: float, s: float, gsd_ref: float, K: int, use_mask: bool):
    """Channels (K,H,W) of edge weights (K=1: isotropic), mask template, footprint, origin."""
    R, k, W, H, ox, oy, cxq, cyq = _geom(q, th, s, gsd_ref)
    p = (q.pts @ R.T) * k + np.array([ox, oy], np.float32)
    if K > 1:
        phi = (q.ori + math.radians(th)) % math.pi
        b = np.floor(phi / (math.pi / K)).astype(int) % K
        tw = np.stack([_splat(p[b == c], q.w[b == c], H, W) for c in range(K)])
    else:
        tw = _splat(p, q.w, H, W)[None]
    a = k * q.gsd
    A = np.array([[a * R[0, 0], a * R[0, 1], 0], [a * R[1, 0], a * R[1, 1], 0]], np.float32)
    A[:, 2] = np.array([ox, oy]) - A[:, :2] @ np.array([cxq, cyq], np.float32)
    tm = cv2.warpAffine(q.mask, A, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.0) if use_mask else None
    tv = cv2.warpAffine(q.valid.astype(np.float32), A, (W, H), flags=cv2.INTER_NEAREST, borderValue=0.0)
    return tw * tv[None], tm, tv, (ox, oy)


# ----------------------------------------------------------------------------
# masked NCC over stacked channels
# ----------------------------------------------------------------------------


class _Region:
    """Spectra of one reference region (cached across all (theta, s))."""

    def __init__(self, chans: torch.Tensor, valid: torch.Tensor):
        # chans: (C,H,W) float32, valid: (H,W)
        self.H, self.W = chans.shape[-2:]
        self.C = chans.shape[0]
        chans = chans * valid[None]
        self.F = torch.fft.rfft2(chans)                          # (C, H, W/2+1)
        self.F1 = torch.fft.rfft2(chans.sum(0))                   # sum_k S_k
        self.F2 = torch.fft.rfft2((chans * chans).sum(0))         # sum_k S_k^2
        self.FV = torch.fft.rfft2(valid)


def _pad(arrs, H, W, dev):
    out = torch.zeros((len(arrs),) + arrs[0].shape[:-2] + (H, W), device=dev, dtype=torch.float32)
    for i, a in enumerate(arrs):
        out[i, ..., : a.shape[-2], : a.shape[-1]] = torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    return out


def _ncc_stack(reg: _Region, T: List[np.ndarray], V: List[np.ndarray], hmax, wmax, dev, eps, min_overlap=0.5):
    """T: list of (C,h,w) templates (already restricted to their footprints); V: (h,w) footprints."""
    H, W, C = reg.H, reg.W, reg.C
    Tt = _pad(T, H, W, dev)                                    # (B,C,H,W)
    Vt = _pad(V, H, W, dev)                                    # (B,H,W)
    FT = torch.conj(torch.fft.rfft2(Tt))                       # (B,C,...)
    FTs = torch.conj(torch.fft.rfft2(Tt.sum(1)))
    FT2 = torch.conj(torch.fft.rfft2((Tt * Tt).sum(1)))
    FV = torch.conj(torch.fft.rfft2(Vt))
    del Tt
    sl = (slice(None), slice(0, H - hmax + 1), slice(0, W - wmax + 1))
    ic = lambda X: torch.fft.irfft2(X, s=(H, W))[sl]
    A = ic((FT * reg.F[None]).sum(1))
    SS = ic(FV * reg.F1[None]); SS2 = ic(FV * reg.F2[None])
    No1 = ic(FV * reg.FV[None]).clamp_min(1.0)                 # overlap (pixels)
    No = No1 * C                                               # stacked domain size
    ST = ic(FTs * reg.FV[None]); ST2 = ic(FT2 * reg.FV[None])
    cov = A - ST * SS / No
    vT = (ST2 - ST * ST / No).clamp_min(1e-9)
    vS = (SS2 - SS * SS / No).clamp_min(0)
    out = cov / torch.sqrt(vT * (vS + No * eps * eps))
    N = torch.tensor([float(v.sum()) + 1e-6 for v in V], device=dev)[:, None, None]
    return torch.where(No1 >= min_overlap * N, out, torch.full_like(out, -1e3))


# ----------------------------------------------------------------------------
# search
# ----------------------------------------------------------------------------


def search(q: QueryStructure, ref: RefMaps, center_uv: Optional[tuple] = None,
           radius_m: Optional[float] = None, cfg: Optional[SearchConfig] = None,
           keep_map: bool = False) -> SearchResult:
    cfg = cfg or SearchConfig()
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    combos = [(float(th), float(s)) for th in cfg.thetas_deg for s in cfg.scales]
    oriented = bool(cfg.oriented and ref.Gk is not None and q.ori is not None)
    K = ref.Gk.shape[0] if oriented else 1
    geo = [_geom(q, th, s, ref.gsd) for th, s in combos]
    hmax = max(g[3] for g in geo); wmax = max(g[2] for g in geo)
    ox_all = np.array([g[4] for g in geo]); oy_all = np.array([g[5] for g in geo])
    Hs, Ws = ref.G.shape

    if center_uv is None:
        cu0, cu1, cv0, cv1 = 0, Ws, 0, Hs
    else:
        r = radius_m / ref.gsd
        cu0, cu1 = int(center_uv[0] - r), int(center_uv[0] + r) + 1
        cv0, cv1 = int(center_uv[1] - r), int(center_uv[1] + r) + 1

    pri = np.array([-((th - cfg.theta0_deg) ** 2) / (2 * cfg.sigma_theta_deg ** 2)
                    - (math.log(s) - math.log(cfg.s0)) ** 2 / (2 * cfg.sigma_logs ** 2)
                    for th, s in combos], np.float32) * 0.01
    ref_valid = ref.extra_valid if hasattr(ref, "extra_valid") else None

    step = max(256, cfg.max_tile - max(hmax, wmax))
    pieces = []
    for tv in range(cv0, cv1, step):
        for tu in range(cu0, cu1, step):
            wv1, wu1 = min(tv + step, cv1), min(tu + step, cu1)
            ru0 = int(math.floor(tu - ox_all.max())); rv0 = int(math.floor(tv - oy_all.max()))
            ru1 = int(math.ceil(wu1 - ox_all.min())) + wmax; rv1 = int(math.ceil(wv1 - oy_all.min())) + hmax
            Hr, Wr = rv1 - rv0, ru1 - ru0
            a0, a1 = max(rv0, 0), min(rv1, Hs); b0, b1 = max(ru0, 0), min(ru1, Ws)
            if a1 <= a0 or b1 <= b0:
                continue
            def crop(arr):
                out = np.zeros(arr.shape[:-2] + (Hr, Wr), np.float32)
                out[..., a0 - rv0:a1 - rv0, b0 - ru0:b1 - ru0] = arr[..., a0:a1, b0:b1]
                return torch.from_numpy(out).to(dev)
            vr = np.zeros((Hr, Wr), np.float32); vr[a0 - rv0:a1 - rv0, b0 - ru0:b1 - ru0] = 1.0
            Vs = torch.from_numpy(vr).to(dev)
            regE = _Region(crop(ref.Gk if oriented else ref.G[None]), Vs) if cfg.use_edge else None
            regM = _Region(crop(ref.M[None]), Vs) if cfg.use_mask else None
            nu, nv = wu1 - tu, wv1 - tv
            smax = torch.full((nv, nu), -1e9, device=dev); sarg = torch.zeros((nv, nu), dtype=torch.long, device=dev)
            for b0i in range(0, len(combos), cfg.batch):
                idx = list(range(b0i, min(b0i + cfg.batch, len(combos))))
                tpl = [_template(q, *combos[i], ref.gsd, K, cfg.use_mask) for i in idx]   # lazy: only this batch
                V = [t[2] for t in tpl]
                tot = None
                if cfg.use_edge:
                    tot = _ncc_stack(regE, [t[0] for t in tpl], V, hmax, wmax, dev, eps=0.05)
                if cfg.use_mask:
                    cm = _ncc_stack(regM, [(t[1] * t[2])[None] for t in tpl], V, hmax, wmax, dev, eps=0.1)
                    tot = cfg.alpha * cm if tot is None else tot + cfg.alpha * cm
                for j, i in enumerate(idx):
                    ox, oy = tpl[j][3]
                    su = int(round(tu - ox)) - ru0; sv = int(round(tv - oy)) - rv0
                    slc = tot[j, sv:sv + nv, su:su + nu]
                    if cfg.zscore:
                        # CFAR-style standardization per (theta, s): smaller templates give
                        # noisier NCC with higher chance maxima; standardizing each score
                        # map by its own mean/std over the window makes scales comparable.
                        ok = slc > -100
                        if ok.sum() > 16:
                            mu = slc[ok].mean(); sd = slc[ok].std().clamp_min(1e-6)
                            slc = torch.where(ok, (slc - mu) / sd, slc)
                    slc = slc + float(pri[i]) * (100.0 if cfg.zscore else 1.0)
                    h_, w_ = slc.shape
                    upd = slc > smax[:h_, :w_]
                    smax[:h_, :w_] = torch.where(upd, slc, smax[:h_, :w_])
                    sarg[:h_, :w_] = torch.where(upd, torch.full_like(sarg[:h_, :w_], i), sarg[:h_, :w_])
                del tot
            pieces.append((smax.cpu().numpy(), sarg.cpu().numpy(), tu, tv))
            del regE, regM
            torch.cuda.empty_cache()

    U0, V0 = cu0, cv0
    Smap = np.full((cv1 - cv0, cu1 - cu0), -1e9, np.float32)
    Amap = np.zeros((cv1 - cv0, cu1 - cu0), np.int64)
    for sm, sa, tu, tv in pieces:
        Smap[tv - V0:tv - V0 + sm.shape[0], tu - U0:tu - U0 + sm.shape[1]] = sm
        Amap[tv - V0:tv - V0 + sa.shape[0], tu - U0:tu - U0 + sa.shape[1]] = sa
    if center_uv is not None:
        yy, xx = np.mgrid[V0:cv1, U0:cu1]
        Smap[(xx - center_uv[0]) ** 2 + (yy - center_uv[1]) ** 2 > (radius_m / ref.gsd) ** 2] = -1e9
    peaks = _nms_peaks(Smap, Amap, combos, U0, V0, cfg.topk, cfg.nms_m / ref.gsd)
    second = -1e9
    if peaks:
        yy, xx = np.mgrid[V0:cv1, U0:cu1]
        far = (xx - peaks[0].u) ** 2 + (yy - peaks[0].v) ** 2 > (cfg.second_peak_excl_m / ref.gsd) ** 2
        if far.any():
            second = float(Smap[far].max())
    return SearchResult(peaks=peaks, second_score=second, score_map=Smap if keep_map else None,
                        region_origin=(U0, V0), extra={"oriented": oriented, "K": K})


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
        S[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1] = -1e9
    return out
