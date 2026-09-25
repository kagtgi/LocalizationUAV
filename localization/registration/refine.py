"""Continuous refinement of J(T) and Laplace pose covariance (§4.7).

Starting from a grid peak, maximize the smooth objective over
p = (u, v, theta, log s) with bilinear sampling of the reference maps, then
approximate the posterior of the pose by a Gaussian whose precision is the
negative Hessian of the (tempered) log-objective at the optimum:

    Sigma_p = ( -N_eff / tau * Hess_p J(p*) )^-1        (Proposition 3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .fft_search import Peak, SearchConfig
from .structure import QueryStructure, RefMaps


@dataclass
class RefineResult:
    u: float
    v: float
    theta_deg: float
    s: float
    J: float
    cov_uv_m2: np.ndarray    # 2x2 position covariance, metres^2
    sigma_pos_m: float       # sqrt(trace) of cov_uv_m2
    hess_ok: bool
    n_iter: int


def _crop(ref: RefMaps, u: float, v: float, half: int, blur_mask_px: float):
    Hs, Ws = ref.G.shape
    u0, v0 = int(u) - half, int(v) - half
    G = np.zeros((2 * half, 2 * half), np.float32); M = np.zeros_like(G)
    a0, a1 = max(v0, 0), min(v0 + 2 * half, Hs); b0, b1 = max(u0, 0), min(u0 + 2 * half, Ws)
    if a1 > a0 and b1 > b0:
        G[a0 - v0 : a1 - v0, b0 - u0 : b1 - u0] = ref.G[a0:a1, b0:b1]
        M[a0 - v0 : a1 - v0, b0 - u0 : b1 - u0] = ref.M[a0:a1, b0:b1]
    if blur_mask_px > 0:
        M = cv2.GaussianBlur(M, (0, 0), blur_mask_px)
    return G, M, u0, v0


def refine(
    q: QueryStructure,
    ref: RefMaps,
    peak: Peak,
    cfg: Optional[SearchConfig] = None,
    iters: int = 150,
    tau: float = 1.0,
    mask_stride: int = 4,
    device: str = "cuda",
) -> RefineResult:
    cfg = cfg or SearchConfig()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    hq, wq = q.mask.shape
    ext = math.hypot(hq, wq) * q.gsd / ref.gsd * max(cfg.scales) / 2 + 40
    half = int(ext) + 2
    G, M, u0, v0 = _crop(ref, peak.u, peak.v, half, 1.0 / ref.gsd)
    Gt = torch.from_numpy(G).to(dev)[None, None]
    Mt = torch.from_numpy(M).to(dev)[None, None]
    Hc, Wc = G.shape

    pts = torch.from_numpy(q.pts).to(dev)
    w = torch.from_numpy(q.w).to(dev)
    wsum = w.sum() + 1e-6
    ys, xs = np.mgrid[0:hq:mask_stride, 0:wq:mask_stride]
    mv = q.mask[ys, xs]
    keep = mv != 0
    mpts = np.stack([(xs[keep] - (wq - 1) / 2) * q.gsd, (ys[keep] - (hq - 1) / 2) * q.gsd], 1).astype(np.float32)
    mpts = torch.from_numpy(mpts).to(dev)
    mval = torch.from_numpy(mv[keep].astype(np.float32)).to(dev)
    msum = mval.abs().sum() + 1e-6

    def sample(img, P):
        gx = P[:, 0] / (Wc - 1) * 2 - 1
        gy = P[:, 1] / (Hc - 1) * 2 - 1
        g = torch.stack([gx, gy], -1)[None, None]
        return F.grid_sample(img, g, mode="bilinear", align_corners=True, padding_mode="zeros")[0, 0, 0]

    def J_of(p):
        u, v, th, ls = p[0], p[1], p[2], p[3]
        s = torch.exp(ls)
        c, sn = torch.cos(th), torch.sin(th)
        k = s / ref.gsd
        def tr(X):
            x = (c * X[:, 0] - sn * X[:, 1]) * k + (u - u0)
            y = (sn * X[:, 0] + c * X[:, 1]) * k + (v - v0)
            return torch.stack([x, y], 1)
        J = torch.zeros((), device=dev)
        if cfg.use_edge:
            J = J + (w * sample(Gt, tr(pts))).sum() / wsum
        if cfg.use_mask:
            J = J + cfg.alpha * (mval * sample(Mt, tr(mpts))).sum() / msum
        th_d = th * 180 / math.pi
        J = J - 0.01 * ((th_d - cfg.theta0_deg) ** 2 / (2 * cfg.sigma_theta_deg ** 2)
                        + (ls - math.log(cfg.s0)) ** 2 / (2 * cfg.sigma_logs ** 2))
        return J

    p = torch.tensor([peak.u, peak.v, math.radians(peak.theta_deg), math.log(peak.s)],
                     device=dev, dtype=torch.float32, requires_grad=True)
    # parameter scaling: pixels ~ 1, radians ~ 0.01, log-scale ~ 0.01
    scale = torch.tensor([1.0, 1.0, 0.01, 0.01], device=dev)
    z = torch.zeros(4, device=dev, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.3)
    base = p.detach().clone()
    best = (-1e9, base.clone())
    n = 0
    for n in range(iters):
        opt.zero_grad()
        val = J_of(base + z * scale)
        (-val).backward()
        opt.step()
        if float(val) > best[0]:
            best = (float(val), (base + z * scale).detach().clone())
    pbest = best[1]

    # Laplace approximation
    hess_ok = True
    try:
        H = torch.autograd.functional.hessian(J_of, pbest).detach().cpu().double().numpy()
        n_eff = float(len(q.pts))
        prec = -H * n_eff / tau
        cov = np.linalg.inv(prec + np.eye(4) * 1e-9)
        cov_uv = cov[:2, :2] * (ref.gsd ** 2)
        if not np.all(np.isfinite(cov_uv)) or np.any(np.linalg.eigvalsh(cov_uv) <= 0):
            raise ValueError("indefinite")
    except Exception:
        hess_ok = False
        cov_uv = np.eye(2) * 1e6
    pb = pbest.cpu().numpy()
    return RefineResult(
        u=float(pb[0]), v=float(pb[1]), theta_deg=math.degrees(float(pb[2])), s=math.exp(float(pb[3])),
        J=best[0], cov_uv_m2=cov_uv, sigma_pos_m=float(math.sqrt(max(np.trace(cov_uv), 0))),
        hess_ok=hess_ok, n_iter=n + 1,
    )
