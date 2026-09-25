#!/usr/bin/env python
"""StructReg on AnyVisLoc (E11-E12, zero-shot), official 'known attitude' regime.

Each sample carries K, distortion, pose_c2w and xyz in a scene-local metric
frame (x = map col direction, y = map row direction, official convention
col=(x-ox)/res, row=(y-oy)/res). With the attitude (roll/pitch/yaw) and the
relative altitude known -- AnyVisLoc's pose-prior setting -- the image is
rectified to the ground plane by the plane-induced homography, giving a
metric, map-aligned orthophoto of the footprint; the camera's horizontal
position is then the unknown translation of StructReg. Error is the official
horizontal distance ||(x,y)_pred - (x,y)_gt||.

Pitch buckets (off-nadir angle): near-nadir <15, mild 15-45, oblique 45-70,
extreme >70 deg.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization.registration.structure import EnsembleConfig, query_structure, reference_maps, resample  # noqa
from localization.registration.fft_search import SearchConfig, search  # noqa
from localization.registration.refine import refine  # noqa

SEG_GSD = 0.3
Image.MAX_IMAGE_PIXELS = None


def load_model(path, device):
    from localization.segmentation.model import load_model as lm
    return lm(path, device=device, num_classes=2, pretrained=False).to(device).eval()


def soft_mask(img, model, device, batch=12):
    from localization.registration.seg import soft_mask as sm
    return sm(img, model, device, batch=batch)


def off_nadir_deg(c2w):
    """Angle between the optical axis and the downward vertical."""
    z = c2w[:3, 2]                       # optical axis in world
    # local frame: z up (rel_alt positive); down = (0,0,-1)
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(z[2]) / (np.linalg.norm(z) + 1e-12)))))


def rectify(img, K, dist, c2w, z_ground, gsd, max_range_m):
    """Ground-plane orthophoto centred on the camera nadir point.

    Output pixel (i, j) <-> local ground point (x, y) = C_xy + ((j - c) gsd, (i - c) gsd).
    Returns ortho RGB and footprint mask.
    """
    C = c2w[:3, 3].astype(np.float64)
    R = c2w[:3, :3].astype(np.float64)
    h_rel = C[2] - z_ground
    n = int(2 * max_range_m / gsd) | 1
    c = n // 2
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    X = C[0] + (jj - c) * gsd
    Y = C[1] + (ii - c) * gsd
    P = np.stack([X - C[0], Y - C[1], np.full_like(X, z_ground - C[2])], -1)   # world rays
    Pc = P @ R                            # world->camera: R^T p  (row-vector form)
    zc = Pc[..., 2]
    ok = zc > 1e-3
    u = K[0, 0] * Pc[..., 0] / np.where(ok, zc, 1) + K[0, 2]
    v = K[1, 1] * Pc[..., 1] / np.where(ok, zc, 1) + K[1, 2]
    H, W = img.shape[:2]
    ok &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    und = cv2.undistort(img, K, dist)
    ortho = cv2.remap(und, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ortho[~ok] = 0
    return ortho, ok.astype(np.uint8), h_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/anyvisloc")
    ap.add_argument("--scenes", nargs="+", default=[f"{i:02d}" for i in range(10, 25)])
    ap.add_argument("--ref", default="satellite", choices=["satellite", "aerial"])
    ap.add_argument("--cache", default="cache/avl")
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--gsd", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-range", type=float, default=150.0)
    ap.add_argument("--theta-range", type=float, default=10.0)
    ap.add_argument("--out", default="results/reg/avl.csv")
    args = ap.parse_args()
    device = torch.device("cuda")
    model = load_model(args.model, device)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out)["sample_id"]) if out.exists() else set()
    cfg = SearchConfig(thetas_deg=tuple(np.arange(-args.theta_range, args.theta_range + 1e-6, 2.5)),
                       scales=(0.9, 0.95, 1.0, 1.05, 1.1), topk=5, device="cuda")
    for sc in args.scenes:
        sd = Path(args.root) / f"Scene_{sc}"
        if not sd.exists():
            print("missing", sd); continue
        refj = json.load(open(next(sd.glob("*_reference.json"))))["modes"][args.ref]
        res_m = float(refj["map_resolution"][0]); ox, oy = map(float, refj["map_origin_local"])
        cdir = Path(args.cache) / f"{sc}_{args.ref}"; cdir.mkdir(parents=True, exist_ok=True)
        pf = cdir / "prob.png"
        if not pf.exists():
            m = Image.open(sd / refj["map_path"]).convert("RGB")
            f = res_m / SEG_GSD
            m = m.resize((max(1, int(m.width * f)), max(1, int(m.height * f))), Image.BILINEAR)
            prob = soft_mask(m, model, device)
            cv2.imwrite(str(pf), (prob * 255).astype(np.uint8))
        prob = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        pw = resample(prob, SEG_GSD, args.gsd)
        ref = reference_maps(pw, args.gsd)
        # ground height: median of the scene's DSM in the local frame
        try:
            dsm = np.load(sd / refj["dsm_path"]).astype(np.float32)
            zg_scene = float(np.nanmedian(dsm[np.isfinite(dsm)]))
        except Exception:
            zg_scene = 0.0
        files = sorted(sd.glob("L*_*.npz"))
        if args.n and len(files) > args.n:
            files = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, args.n).round().astype(int))]
        rows = []
        for fp in files:
            sid = fp.stem
            if sid in done:
                continue
            z = np.load(fp, allow_pickle=True)
            img = z["image"]; K = z["K"].astype(np.float64); dist = z["dist"].astype(np.float64)
            c2w = z["pose_c2w"].astype(np.float64); xyz = z["xyz"]; eul = z["euler_deg"]
            ond = off_nadir_deg(c2w)
            # the rel_alt convention: ground plane at z=0 unless the DSM says otherwise
            zg = 0.0 if abs(zg_scene) > 0.5 * abs(float(xyz[2])) else zg_scene
            t0 = time.time()
            ortho, valid, h_rel = rectify(img, K, dist, c2w, zg, SEG_GSD, args.max_range)
            qprob = soft_mask(Image.fromarray(ortho), model, device) * valid
            qpw = resample(qprob, SEG_GSD, args.gsd)
            vw = (resample(valid.astype(np.float32), SEG_GSD, args.gsd) > 0.5).astype(np.uint8)
            q = query_structure(qpw, args.gsd, valid=vw, ens=EnsembleConfig(dp_tol_m=(0.25, 0.5, 1.0)))
            t_q = time.time() - t0
            gx, gy = (float(xyz[0]) - ox) / res_m * res_m / args.gsd, (float(xyz[1]) - oy) / args.gsd
            gu, gv = (float(xyz[0]) - ox) / args.gsd, (float(xyz[1]) - oy) / args.gsd
            bucket = "near" if ond < 15 else "mild" if ond < 45 else "oblique" if ond < 70 else "extreme"
            rec = dict(scene=sc, sample_id=sid, off_nadir=ond, bucket=bucket, rel_alt=float(xyz[2]) - zg,
                       roll=float(eul[0]), pitch=float(eul[1]), yaw=float(eul[2]),
                       n_buildings=q.n_buildings, persistence=q.mean_persistence, t_query=t_q, ref=args.ref)
            if len(q.pts) < 20:
                rec.update(status="no_structure", err_m=np.nan)
                rows.append(rec); continue
            t1 = time.time()
            sr = search(q, ref, center_uv=None, radius_m=None, cfg=cfg)
            if not sr.peaks:
                rec.update(status="no_peak", err_m=np.nan); rows.append(rec); continue
            pk = sr.peaks[0]
            rr = refine(q, ref, pk, cfg)
            rec.update(status="ok", pred_x=rr.u * args.gsd + ox, pred_y=rr.v * args.gsd + oy,
                       err_m=math.hypot(rr.u - gu, rr.v - gv) * args.gsd,
                       err_grid_m=math.hypot(pk.u - gu, pk.v - gv) * args.gsd,
                       J1=pk.score, peak_ratio=pk.score - sr.second_score, sigma_pos_m=rr.sigma_pos_m,
                       theta=rr.theta_deg, s=rr.s, t_opt=time.time() - t1)
            rows.append(rec)
            if len(rows) >= 20:
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False); rows = []
                print(sc, sid, f"err={rec['err_m']:.1f} off-nadir={ond:.0f}", flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    d = pd.read_csv(out)
    for t in (1, 3, 5, 10, 20):
        d[f"S@{t}"] = (d.err_m <= t).astype(float)
    print(d.groupby("bucket")[["err_m", "S@5", "S@10", "S@20"]].agg(["median", "mean", "count"]).to_string())


if __name__ == "__main__":
    main()
