#!/usr/bin/env python
"""StructReg on World-UAV / UAV-GeoLoc 'Rot' test set (E9-E10, zero-shot).

The Rot split renders one trajectory at 3 heights x 360/5 headings over one
region. Its satellite DB is a grid of geo-tagged 200x200 crops, which we
mosaic into one continuous map. Heading is UNKNOWN here: the search covers
theta in [0, 360) and scale over the altitude uncertainty, i.e. the full
Sim(2) grid, so rotation robustness is a property of the optimizer, not of a
learned invariance.

Outputs one row per query: metric error, R@1 (nearest DB crop == positive),
semi-positive hit, integrity features, runtimes.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization.registration.structure import EnsembleConfig, query_structure, reference_maps, resample, RefMaps  # noqa
from localization.registration.fft_search import SearchConfig, search  # noqa
from localization.registration.refine import refine  # noqa

SEG_GSD = 0.3


def load_model(path, device):
    from localization.segmentation.model import load_model as lm
    return lm(path, device=device, num_classes=2, pretrained=False).to(device).eval()


def soft_mask(img, model, device, batch=12):
    from localization.registration.seg import soft_mask as sm
    return sm(img, model, device, batch=batch)


def read_db(region: Path):
    rows = []
    for line in open(region / "DB" / "db_postion.txt"):
        p = line.split()
        if len(p) >= 5:
            rows.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return pd.DataFrame(rows, columns=["name", "lon", "lat", "dlon", "dlat"])


def mosaic(region: Path, db: pd.DataFrame):
    """Place every crop at its geo-position. Returns RGB mosaic + geo transform."""
    dlon, dlat = db.dlon.iloc[0], db.dlat.iloc[0]          # deg per px (dlat < 0)
    lon0, lat0 = db.lon.min(), db.lat.max()
    im0 = Image.open(region / "DB" / "img" / db.name.iloc[0])
    cw, ch = im0.size
    xs = np.round((db.lon - lon0) / dlon).astype(int)
    ys = np.round((db.lat - lat0) / dlat).astype(int)
    W, H = xs.max() + cw, ys.max() + ch
    acc = np.zeros((H, W, 3), np.float32); cnt = np.zeros((H, W, 1), np.float32)
    for (n, x, y) in zip(db.name, xs, ys):
        a = np.asarray(Image.open(region / "DB" / "img" / n).convert("RGB"), np.float32)
        acc[y:y + ch, x:x + cw] += a; cnt[y:y + ch, x:x + cw] += 1
    img = (acc / np.maximum(cnt, 1)).astype(np.uint8)
    lat_c = lat0 + dlat * H / 2
    gx = abs(dlon) * 111412.84 * math.cos(math.radians(lat_c))
    gy = abs(dlat) * (111132.954 - 559.822 * math.cos(2 * math.radians(lat_c)))
    geo = dict(lon0=lon0, lat0=lat0, dlon=dlon, dlat=dlat, W=W, H=H, gsd=0.5 * (gx + gy), cw=cw, ch=ch)
    # crop centres (mosaic px) for R@1
    cents = np.stack([xs + cw / 2, ys + ch / 2], 1).astype(np.float32)
    return img, geo, cents


def ll_to_px(lat, lon, geo):
    return (lon - geo["lon0"]) / geo["dlon"], (lat - geo["lat0"]) / geo["dlat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="data/worlduav/Rot/SouthernSuburbs")
    ap.add_argument("--cache", default="cache/wuav")
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--gsd", type=float, default=0.6)
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--theta-step", type=float, default=5.0)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.8, 0.9, 1.0, 1.1, 1.25])
    ap.add_argument("--alt-mode", default="json", choices=["json", "folder"])
    ap.add_argument("--rot-filter", default="")
    ap.add_argument("--out", default="results/reg/wuav_rot.csv")
    args = ap.parse_args()
    region = Path(args.region); cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = load_model(args.model, device)
    db = read_db(region)
    img, geo, cents = mosaic(region, db)
    f = cache / "sat_prob.png"
    if not f.exists():
        up = geo["gsd"] / SEG_GSD
        big = Image.fromarray(img).resize((int(img.shape[1] * up), int(img.shape[0] * up)), Image.BILINEAR)
        prob = soft_mask(big, model, device)
        cv2.imwrite(str(f), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(cache / "sat_rgb.jpg"), img[:, :, ::-1])
    prob = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    pw = resample(prob, SEG_GSD, args.gsd)
    ref = reference_maps(pw, args.gsd)
    k_ref = geo["gsd"] / args.gsd            # mosaic px -> working px
    pos = json.load(open(region / "positive.json"))
    semi = json.load(open(region / "semi_positive.json"))
    cfg = SearchConfig(thetas_deg=tuple(np.arange(0, 360, args.theta_step)), scales=tuple(args.scales),
                       sigma_theta_deg=1e6, sigma_logs=0.3, topk=5, device="cuda")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out)["qid"]) if out.exists() else set()
    rows = []
    folders = sorted((region / "query").iterdir())
    for qf in folders:
        m = re.match(r"height(\d+)_rot(\d+)", qf.name)
        if not m or (args.rot_filter and not re.search(args.rot_filter, qf.name)):
            continue
        hgt, rot = int(m.group(1)), int(m.group(2))
        cam = json.load(open(qf / f"{qf.name}.json"))
        frames = sorted((qf / "footage").glob("*.jp*g"))
        for fi, fp in enumerate(frames):
            if fi % args.frame_step:
                continue
            qid = fp.stem
            if qid in done:
                continue
            idx = int(qid.split("_")[-1])
            cf = cam["cameraFrames"][min(idx, len(cam["cameraFrames"]) - 1)]
            alt = float(cf["coordinate"]["altitude"]) if args.alt_mode == "json" else float(hgt)
            fov = math.radians(cf.get("fovVertical", 30.0))
            W_img = cam["width"]
            gsd_q = 2 * alt * math.tan(fov / 2) / cam["height"]
            t0 = time.time()
            im = Image.open(fp).convert("RGB")
            fct = gsd_q / SEG_GSD
            im = im.resize((max(1, int(im.width * fct)), max(1, int(im.height * fct))), Image.BILINEAR)
            qprob = soft_mask(im, model, device)
            qpw = resample(qprob, SEG_GSD, args.gsd)
            q = query_structure(qpw, args.gsd, ens=EnsembleConfig(dp_tol_m=(0.5, 1.0)))
            t_seg = time.time() - t0
            gx, gy = ll_to_px(cf["coordinate"]["latitude"], cf["coordinate"]["longitude"], geo)
            gu, gv = gx / k_ref, gy / k_ref
            rec = dict(qid=qid, height=hgt, rot=rot, alt=alt, frame=idx, n_buildings=q.n_buildings,
                       gt_u=gu, gt_v=gv, persistence=q.mean_persistence, t_seg=t_seg)
            if len(q.pts) < 20:
                rec.update(status="no_structure", err_m=np.nan, r1=0, r1_semi=0)
                rows.append(rec); continue
            t1 = time.time()
            res = search(q, ref, center_uv=None, radius_m=None, cfg=cfg)
            pk = res.peaks[0]
            rr = refine(q, ref, pk, cfg)
            t_opt = time.time() - t1
            pu, pv = rr.u, rr.v
            # nearest DB crop centre -> R@1 against positive / semi-positive
            d = np.hypot(cents[:, 0] / k_ref - pu, cents[:, 1] / k_ref - pv)
            order = np.argsort(d)
            key = f"{idx:02d}"
            posn, semin = set(pos.get(key, [])), set(semi.get(key, [])) | set(pos.get(key, []))
            names = db.name.values
            rec.update(status="ok", pred_u=pu, pred_v=pv, err_m=math.hypot(pu - gu, pv - gv) * args.gsd,
                       err_grid_m=math.hypot(pk.u - gu, pk.v - gv) * args.gsd,
                       theta=rr.theta_deg, s=rr.s, J1=pk.score, peak_ratio=pk.score - res.second_score,
                       sigma_pos_m=rr.sigma_pos_m,
                       r1=int(names[order[0]] in posn), r1_semi=int(names[order[0]] in semin),
                       r5=int(any(names[o] in posn for o in order[:5])), t_opt=t_opt)
            rows.append(rec)
            if len(rows) >= 20:
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False); rows = []
                print(qf.name, qid, f"err={rec['err_m']:.1f}m r1={rec['r1']}", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    d = pd.read_csv(out)
    print(d.groupby("rot")[["err_m", "r1"]].agg(["median", "mean"]).to_string())


if __name__ == "__main__":
    main()
