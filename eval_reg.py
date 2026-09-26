#!/usr/bin/env python
"""StructReg evaluation on UAV-VisLoc (resumable; one CSV row per query).

Stages
------
satcache  segment each satellite GeoTIFF once -> building probability (uint8,
          native GSD) + working maps (G, M at --gsd) + building polygons and
          vertex-coupled MFCA signatures for verification.
uavcache  select queries, rotate north-up by IMU yaw, resample to the
          segmentation GSD with the calibrated altitude->GSD factor, segment,
          cache probability + footprint.
calib     estimate the altitude->GSD factor k (GSD = k * height) on the
          CALIBRATION sites only, by a wide scale search around GT.
run       prior-window / global structural pose optimization; logs pose,
          error, integrity features, runtimes.

Everything is keyed by site and query id, so reruns skip finished work.
"""
from __future__ import annotations

import argparse
import json
import zlib
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization.io.bounds import load_satellite_bounds  # noqa: E402
from localization.io.dataset import VisLocFlight, load_flight_metadata  # noqa: E402
from localization.registration.structure import (  # noqa: E402
    EnsembleConfig, query_structure, reference_maps, reference_polygons, resample)
from localization.registration.fft_search import SearchConfig, search  # noqa: E402
from localization.registration.refine import refine  # noqa: E402
from localization.registration.verify import ShapeVerifier, signatures, polygon_areas  # noqa: E402
from localization.registration import lines as LN  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SEG_GSD = 0.3            # the Mask R-CNN operates at ~0.3 m/px (satellite native)


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------


def site_geo(root: Path, site: str):
    fl = VisLocFlight(site, root)
    b = load_satellite_bounds(fl.satellite_tif.name, str(fl.bounds_csv))
    with Image.open(fl.satellite_tif) as im:
        W, H = im.size
    lat_c = 0.5 * (b["LT_lat"] + b["RB_lat"])
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(lat_c))
    m_per_deg_lon = 111412.84 * math.cos(math.radians(lat_c))
    gx = (b["RB_lon"] - b["LT_lon"]) * m_per_deg_lon / (W - 1)
    gy = (b["LT_lat"] - b["RB_lat"]) * m_per_deg_lat / (H - 1)
    return dict(bounds=b, W=W, H=H, gsd_x=gx, gsd_y=gy, gsd=0.5 * (gx + gy))


def latlon_to_px_f(lat, lon, g):
    b = g["bounds"]
    x = (lon - b["LT_lon"]) / (b["RB_lon"] - b["LT_lon"]) * (g["W"] - 1)
    y = (b["LT_lat"] - lat) / (b["LT_lat"] - b["RB_lat"]) * (g["H"] - 1)
    return x, y


def cache_dir(args, *parts):
    p = Path(args.cache).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_model(args, device):
    from localization.segmentation.model import load_model as lm
    return lm(args.model, device=device, num_classes=2, pretrained=False).to(device).eval()


def soft_mask(img_rgb: Image.Image, model, device, batch=12):
    from localization.registration.seg import soft_mask as sm
    return sm(img_rgb, model, device, batch=batch)


# ----------------------------------------------------------------------------
# satellite cache
# ----------------------------------------------------------------------------


def stage_satcache(args):
    device = torch.device("cuda")
    model = None
    for site in args.sites:
        out = cache_dir(args, "sat", site)
        if args.frontend == "lines":
            f = out / f"lines_or{LN.LineConfig().K}_{args.gsd:.2f}.npz"
            if f.exists():
                print(f"[sat {site}] lines cached"); continue
            t0 = time.time()
            g = site_geo(Path(args.root), site)
            gray = np.asarray(Image.open(VisLocFlight(site, Path(args.root)).satellite_tif).convert("L"))
            nominal, _ = LN.line_maps(gray, g["gsd"], args.gsd, persistence=False)
            ref = LN.reference_from_lines(nominal, args.gsd, sigma_m=args.sigma_m, K=LN.LineConfig().K)
            np.savez_compressed(f, G=ref.G.astype(np.float16), Gk=ref.Gk, lines=nominal.astype(np.uint8))
            print(f"[sat {site}] lines {nominal.shape}, {int(nominal.sum())} line px in {time.time()-t0:.0f}s", flush=True)
            continue
        if (out / f"work_{args.gsd:.2f}.npz").exists() and (out / "sigs.npz").exists():
            print(f"[sat {site}] cached"); continue
        g = site_geo(Path(args.root), site)
        probf = out / "prob_native.png"
        if not probf.exists():
            model = model or load_model(args, device)
            t0 = time.time()
            img = Image.open(VisLocFlight(site, Path(args.root)).satellite_tif).convert("RGB")
            # segment at SEG_GSD (native is ~0.3 m already for UAV-VisLoc)
            f = g["gsd"] / SEG_GSD
            if abs(f - 1) > 0.05:
                img = img.resize((int(img.width * f), int(img.height * f)), Image.BILINEAR)
            from localization.registration.seg import soft_mask as sm
            prob8 = sm(img, model, device, batch=args.batch, as_uint8=True)   # memory-lean (20 GB cap)
            del img
            cv2.imwrite(str(probf), prob8)
            json.dump({"seg_gsd": g["gsd"] / f if abs(f - 1) > 0.05 else g["gsd"], "secs": time.time() - t0,
                       "shape": list(prob8.shape)}, open(out / "prob_meta.json", "w"))
            print(f"[sat {site}] segmented {prob8.shape} in {time.time()-t0:.0f}s", flush=True)
            del prob8
            torch.cuda.empty_cache()
        meta = json.load(open(out / "prob_meta.json"))
        prob8 = cv2.imread(str(probf), cv2.IMREAD_GRAYSCALE)
        pw = resample(prob8, meta["seg_gsd"], args.gsd).astype(np.float32) / 255.0   # resample on uint8
        del prob8
        ref = reference_maps(pw, args.gsd, sigma_m=args.sigma_m)
        np.savez_compressed(out / f"work_{args.gsd:.2f}.npz", G=ref.G.astype(np.float16), M=ref.M.astype(np.int8),
                            prob=(pw * 255).astype(np.uint8))
        polys, cents = reference_polygons(pw, args.gsd)
        polys_m = [p * args.gsd for p in polys]
        sig = signatures(polys_m, coupled=True, workers=args.workers)
        sig0 = signatures(polys_m, coupled=False, workers=args.workers)
        np.savez_compressed(out / "sigs.npz", cents=cents, sig=sig, sig_uncoupled=sig0,
                            area=polygon_areas(polys), gsd=args.gsd)
        print(f"[sat {site}] work maps {ref.G.shape}, {len(polys)} buildings", flush=True)


def stage_satgraph(args):
    """Map line graph (filtered lines, junctions, structural adjacency, CDT) per site, metres."""
    import pickle
    from localization.registration import linegraph as LG
    for site in args.sites:
        f = cache_dir(args, "sat", site) / "graph.pkl"
        if f.exists():
            print(f"[graph {site}] cached"); continue
        t0 = time.time()
        g = site_geo(Path(args.root), site)
        gray = np.asarray(Image.open(VisLocFlight(site, Path(args.root)).satellite_tif).convert("L"))
        segs = LG.filter_lines(LN.segments_m(gray, g["gsd"]))
        del gray
        mg = LG.build_graph(segs, cdt=False)      # CDT of the full map is not needed for verification
        pickle.dump(mg, open(f, "wb"))
        print(f"[graph {site}] {len(segs)} lines, {len(mg.J)} junctions in {time.time()-t0:.0f}s", flush=True)


def query_graph(args, fl, row, k):
    from localization.registration import linegraph as LG
    img, _ = uav_canonical(args, fl, row, k)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return LG.build_graph(LG.filter_lines(LN.segments_m(gray, SEG_GSD, centre=True)), cdt=True)


def load_ref(args, site):
    out = cache_dir(args, "sat", site)
    if args.frontend == "lines":
        z = np.load(out / f"lines_or{LN.LineConfig().K}_{args.gsd:.2f}.npz")
        from localization.registration.structure import RefMaps
        G = z["G"].astype(np.float32)
        return RefMaps(G=G, M=np.zeros((1, 1), np.float32), gsd=args.gsd, Gk=z["Gk"]), None, z["lines"]
    z = np.load(out / f"work_{args.gsd:.2f}.npz")
    from localization.registration.structure import RefMaps
    ref = RefMaps(G=z["G"].astype(np.float32), M=z["M"].astype(np.float32), gsd=args.gsd)
    s = np.load(out / "sigs.npz")
    key = "sig" if args.coupled else "sig_uncoupled"
    ver = ShapeVerifier(s["cents"], s[key], s["area"], args.gsd)
    return ref, ver, z["prob"]


# ----------------------------------------------------------------------------
# queries
# ----------------------------------------------------------------------------


def prior_uv(args, site, fname, gu, gv, radius=None):
    """INS/VO prior centre (working-GSD pixels): GT + offset uniform in a disk of
    radius R*prior_frac. Stable per-query seed, so every stage and every method
    sees the same prior. Returns (cu, cv, rng) -- rng continues for baselines."""
    R = args.radius if radius is None else radius
    rng = np.random.default_rng(zlib.crc32(f"{args.seed}|{site}|{fname}".encode()))
    r = R * args.prior_frac * math.sqrt(rng.uniform()); a = rng.uniform(0, 2 * math.pi)
    return gu + r * math.cos(a) / args.gsd, gv + r * math.sin(a) / args.gsd, rng


def _agl_column(args, site, df):
    """Height above ground = metadata height (above sea level) - DEM median in the
    prior window. Uses the prior centre only, never the ground-truth position."""
    if args.agl != "dem":
        return df["height"].astype(float)
    from localization.io.dem import DEM
    from localization.io.bounds import pixel_to_latlon
    dem = DEM(Path(args.cache).parent / "dem")
    g = site_geo(Path(args.root), site)
    R = args.radius if args.radius > 0 else 1000.0
    out = []
    for _, row in df.iterrows():
        gx, gy = latlon_to_px_f(float(row["lat"]), float(row["lon"]), g)
        cu, cv, _ = prior_uv(args, site, row["filename"], gx * g["gsd"] / args.gsd, gy * g["gsd"] / args.gsd, R)
        plat, plon = pixel_to_latlon(cu * args.gsd / g["gsd"], cv * args.gsd / g["gsd"], g["bounds"], g["W"], g["H"])
        out.append(float(row["height"]) - dem.window_median(plat, plon, R))
    return pd.Series(out, index=df.index)


def select_queries(args, site):
    fl = VisLocFlight(site, Path(args.root))
    df = load_flight_metadata(fl.metadata_csv)
    df = df[df["filename"].str.lower().str.endswith((".jpg", ".jpeg", ".png"))].reset_index(drop=True)
    if args.n and len(df) > args.n:
        idx = np.linspace(0, len(df) - 1, args.n).round().astype(int)
        df = df.iloc[np.unique(idx)].reset_index(drop=True)
    df["height_agl"] = _agl_column(args, site, df)
    return fl, df


def uav_canonical(args, fl, row, k, yaw_noise=0.0):
    """North-up UAV image resampled to SEG_GSD (GSD = k * height above ground) + footprint mask."""
    img = Image.open(fl.drone_image_path(row["filename"])).convert("RGB")
    # GSD = c * AGL / W_px with c = 2 tan(HFOV/2) (camera field-of-view constant; sites use different cameras)
    f = k * float(row.get("height_agl", row["height"])) / img.width / SEG_GSD
    img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
    valid = Image.new("L", img.size, 255)
    yaw = float(row["Phi1"]) + yaw_noise
    img = img.rotate(-yaw, resample=Image.BILINEAR, expand=True)
    valid = valid.rotate(-yaw, resample=Image.NEAREST, expand=True)
    return img, (np.asarray(valid) > 0).astype(np.uint8)


def uav_lines(args, fl, row, k, persistence=True):
    """Training-free query structure: line raster + persistence at the working GSD."""
    img, valid = uav_canonical(args, fl, row, k)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    vw = (resample(valid.astype(np.float32), SEG_GSD, args.gsd) > 0.5).astype(np.uint8)
    nominal, pers = LN.line_maps(gray, SEG_GSD, args.gsd, valid=vw, persistence=persistence)
    vw = cv2.resize(vw, (nominal.shape[1], nominal.shape[0]), interpolation=cv2.INTER_NEAREST)
    return nominal, pers, vw


def uav_prob(args, fl, row, k, model, device, yaw_noise=0.0):
    """North-up UAV building probability at SEG_GSD, plus footprint mask."""
    img = Image.open(fl.drone_image_path(row["filename"])).convert("RGB")
    h = float(row.get("height_agl", row["height"]))
    gsd_raw = k * h / Image.open(fl.drone_image_path(row["filename"])).width
    f = gsd_raw / SEG_GSD
    img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
    valid = Image.new("L", img.size, 255)
    yaw = float(row["Phi1"]) + yaw_noise
    img = img.rotate(-yaw, resample=Image.BILINEAR, expand=True)
    valid = valid.rotate(-yaw, resample=Image.NEAREST, expand=True)
    prob = soft_mask(img, model, device, batch=args.batch)
    return prob, (np.asarray(valid) > 0).astype(np.uint8)


def stage_uavcache(args):
    device = torch.device("cuda")
    model = load_model(args, device) if args.frontend != "lines" else None
    k = json.load(open(Path(args.cache) / "calib.json"))["k"] if args.k is None else args.k
    for site in args.sites:
        fl, df = select_queries(args, site)
        out = cache_dir(args, "uav", site)
        (out / "queries.txt").write_text("\n".join(df["filename"].astype(str)) + "\n")   # for paired M0 runs
        t0 = time.time(); n = 0
        for _, row in df.iterrows():
            f = out / (Path(row["filename"]).stem + ".npz")
            if f.exists():
                continue
            if args.frontend == "lines":
                nominal, pers, vw = uav_lines(args, fl, row, k)
                np.savez_compressed(f, lines=nominal, pers=(pers * 255).astype(np.uint8), valid=vw, k=k)
            else:
                prob, valid = uav_prob(args, fl, row, k, model, device)
                np.savez_compressed(f, prob=(prob * 255).astype(np.uint8), valid=valid, k=k)
            n += 1
        print(f"[uav {site}] {n} new in {time.time()-t0:.0f}s ({len(df)} selected)", flush=True)


def load_query(args, site, fname, persist=True):
    z = np.load(cache_dir(args, "uav", site) / (Path(fname).stem + ".npz"))
    if args.frontend == "lines":
        return LN.query_from_lines(z["lines"], z["pers"].astype(np.float32) / 255.0, args.gsd, z["valid"],
                                   use_persistence=persist)
    prob = z["prob"].astype(np.float32) / 255.0
    valid = z["valid"]
    pw = resample(prob, SEG_GSD, args.gsd)
    vw = (resample(valid.astype(np.float32), SEG_GSD, args.gsd) > 0.5).astype(np.uint8)
    return query_structure(pw, args.gsd, valid=vw, use_persistence=persist,
                           ens=EnsembleConfig(dp_tol_m=(0.5, 1.0, 2.0)))


# ----------------------------------------------------------------------------
# calibration of k (GSD = k * height) on calibration sites
# ----------------------------------------------------------------------------


def stage_calib(args):
    device = torch.device("cuda")
    model = load_model(args, device) if args.frontend != "lines" else None
    k0 = args.k0
    ests = []
    for site in args.calib_sites:
        g = site_geo(Path(args.root), site)
        ref, _, _ = load_ref(args, site)
        fl, df = select_queries(argparse.Namespace(**{**vars(args), "n": args.calib_n}), site)
        for _, row in df.iterrows():
            if args.frontend == "lines":
                nominal, pers, vw = uav_lines(args, fl, row, k0, persistence=False)
                q = LN.query_from_lines(nominal, pers, args.gsd, vw, use_persistence=False)
            else:
                prob, valid = uav_prob(args, fl, row, k0, model, device)
                pw = resample(prob, SEG_GSD, args.gsd)
                vw = (resample(valid.astype(np.float32), SEG_GSD, args.gsd) > 0.5).astype(np.uint8)
                q = query_structure(pw, args.gsd, valid=vw, use_persistence=False)
            if q.n_buildings < 5:
                continue
            gx, gy = latlon_to_px_f(float(row["lat"]), float(row["lon"]), g)
            c = (gx * g["gsd"] / args.gsd, gy * g["gsd"] / args.gsd)
            cfg = SearchConfig(thetas_deg=tuple(np.arange(-6, 6.1, 3.0)),
                               scales=tuple(np.exp(np.linspace(math.log(0.25), math.log(2.0), 31))),
                               sigma_logs=10.0, use_mask=args.frontend != "lines", device="cuda")
            res = search(q, ref, center_uv=c, radius_m=60.0, cfg=cfg)
            if res.peaks:
                pk = res.peaks[0]
                ests.append(dict(site=site, file=row["filename"], s=pk.s, score=pk.score,
                                 ratio=pk.score - res.second_score, nb=q.n_buildings))
                print(site, row["filename"], f"s={pk.s:.3f} score={pk.score:.3f}", flush=True)
    E = pd.DataFrame(ests)
    good = E[E["ratio"] > E["ratio"].median()] if len(E) > 10 else E
    s_med = float(np.median(good["s"]))
    k = k0 * s_med
    json.dump({"k": k, "k0": k0, "s_median": s_med, "n": int(len(good)), "sites": args.calib_sites},
              open(Path(args.cache) / "calib.json", "w"), indent=1)
    E.to_csv(Path(args.cache) / "calib_rows.csv", index=False)
    print("calibrated k =", k, "from", len(good), "frames")


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------


def oracle_query(q, sat_pw, gu, gv, args):
    """Oracle structure (E1): the satellite's own building probability under the
    TRUE footprint (same shape/orientation as the UAV query), so any remaining
    error is due to the matcher/objective, not to UAV segmentation."""
    h, w = q.valid.shape
    u0, v0 = int(round(gu - (w - 1) / 2)), int(round(gv - (h - 1) / 2))
    if args.frontend == "lines":
        H, W = sat_pw.shape
        crop = np.zeros((h, w), np.uint8)
        a0, a1 = max(v0, 0), min(v0 + h, H); b0, b1 = max(u0, 0), min(u0 + w, W)
        if a1 > a0 and b1 > b0:
            crop[a0 - v0:a1 - v0, b0 - u0:b1 - u0] = sat_pw[a0:a1, b0:b1]
        crop = crop * cv2.erode(q.valid.astype(np.uint8), np.ones((7, 7), np.uint8))
        return LN.query_from_lines(crop, np.ones(crop.shape, np.float32), args.gsd, q.valid, use_persistence=False)
    H, W = sat_pw.shape
    crop = np.zeros((h, w), np.float32)
    a0, a1 = max(v0, 0), min(v0 + h, H); b0, b1 = max(u0, 0), min(u0 + w, W)
    if a1 > a0 and b1 > b0:
        crop[a0 - v0:a1 - v0, b0 - u0:b1 - u0] = sat_pw[a0:a1, b0:b1].astype(np.float32) / 255.0
    return query_structure(crop * (q.valid > 0), args.gsd, valid=q.valid, use_persistence=not args.no_persist,
                           ens=EnsembleConfig(dp_tol_m=(0.5, 1.0, 2.0)))


def stage_run(args):
    rng_master = np.random.default_rng(args.seed)
    outcsv = Path(args.out)
    outcsv.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outcsv.exists():
        prev = pd.read_csv(outcsv)
        done = set(zip(prev["site"].astype(str).str.zfill(2), prev["file"]))
    cfg = SearchConfig(thetas_deg=tuple(np.arange(-args.theta_range, args.theta_range + 1e-6, args.theta_step)),
                       scales=tuple(args.scales), alpha=args.alpha, use_edge=not args.no_edge,
                       use_mask=not args.no_mask, topk=args.topk, device="cuda")
    if args.frontend == "lines":
        cfg.use_mask = False      # line structure has no region term
        cfg.oriented = not args.isotropic
        args.gamma = 0.0          # MFCA verification needs closed footprints
    kcal = json.load(open(Path(args.cache) / "calib.json"))["k"] if args.k is None else args.k
    for site in args.sites:
        g = site_geo(Path(args.root), site)
        ref, ver, sat_pw = load_ref(args, site)
        mgraph = mtree = None
        if args.verify_graph:
            import pickle
            from scipy.spatial import cKDTree
            gp = cache_dir(args, "sat", site) / "graph.pkl"
            if gp.exists():
                mgraph = pickle.load(open(gp, "rb"))
                # map graph is in native-GSD metres from the map origin == working-GSD metres
                mtree = cKDTree(mgraph.J) if len(mgraph.J) else None
        fl, df = select_queries(args, site)
        rows = []
        for qi, row in df.iterrows():
            if (site, row["filename"]) in done:
                continue
            fq = cache_dir(args, "uav", site) / (Path(row["filename"]).stem + ".npz")
            if not fq.exists():
                continue
            t0 = time.time()
            q = load_query(args, site, row["filename"], persist=not args.no_persist)
            t_q = time.time() - t0
            gx, gy = latlon_to_px_f(float(row["lat"]), float(row["lon"]), g)
            gu, gv = gx * g["gsd"] / args.gsd, gy * g["gsd"] / args.gsd
            # stable per-query prior (same generator as caching/DEM lookup): paired across modes
            pcu, pcv, rng = prior_uv(args, site, row["filename"], gu, gv)
            q_uav_nb = q.n_buildings
            if args.oracle:
                q = oracle_query(q, sat_pw, gu, gv, args)
            if args.local_radius > 0:
                center, rad = (gu, gv), args.local_radius
                cu, cv = gu, gv
            elif args.radius > 0:
                # prior: GT + offset uniform in a disk of radius radius*prior_frac (INS/VO drift model)
                cu, cv = pcu, pcv
                center, rad = (cu, cv), args.radius
            else:
                center, rad = None, None
                cu, cv = ref.G.shape[1] / 2, ref.G.shape[0] / 2
            rec = dict(site=site, file=row["filename"], height=float(row["height"]), height_agl=float(row["height_agl"]), yaw=float(row["Phi1"]),
                       n_buildings=q.n_buildings, n_pts=len(q.pts), persistence=q.mean_persistence, coverage=q.coverage,
                       gt_u=gu, gt_v=gv, prior_u=cu, prior_v=cv, radius=args.radius,
                       prior_err_m=math.hypot(cu - gu, cv - gv) * args.gsd, uav_n_buildings=q_uav_nb,
                       mode=args.frontend + "_" + ("oracle" if args.oracle else "uav") + (f"_local{int(args.local_radius)}" if args.local_radius > 0 else ""))
            if rad:
                rr_ = rad * math.sqrt(rng.uniform()); aa_ = rng.uniform(0, 2 * math.pi)
                ru_, rv_ = center[0] + rr_ * math.cos(aa_) / args.gsd, center[1] + rr_ * math.sin(aa_) / args.gsd
                rec["rand_err_m"] = math.hypot(ru_ - gu, rv_ - gv) * args.gsd
            if q.n_buildings == 0 or len(q.pts) < 20:
                rec.update(pred_u=cu, pred_v=cv, err_m=rec["prior_err_m"], err_grid_m=rec["prior_err_m"],
                           J1=0, peak_ratio=0, A=0, sigma_pos_m=1e4, log_sigma_pos=math.log(1e4), status="no_structure",
                           t_search=0, t_refine=0, t_verify=0, t_query=t_q, spread=0)
                rows.append(rec); continue
            t1 = time.time()
            res = search(q, ref, center_uv=center, radius_m=rad, cfg=cfg)
            torch.cuda.synchronize(); t_s = time.time() - t1
            # Ekeland shape verification of top-k peaks
            t2 = time.time()
            qsig = signatures(q.polygons, coupled=args.coupled) if args.gamma > 0 else None
            best, bestval, Avals, vers = None, -1e9, [], []
            if args.verify_graph and mgraph is not None:
                from localization.registration import linegraph as LG
                qg = query_graph(args, fl, row, kcal)
            for pk in res.peaks:
                A = ver.agreement(q.polygons, qsig, pk.u, pk.v, pk.theta_deg, pk.s)["A"] if args.gamma > 0 else 0.0
                if args.verify_graph and mgraph is not None:
                    vr = LG.verify(qg, mgraph, mtree, math.radians(pk.theta_deg), pk.s,
                                   np.array([pk.u, pk.v]) * args.gsd, use_ekeland=not args.no_ekeland)
                    vers.append(vr); A = vr.score
                Avals.append(A)
                val = pk.score + (args.gamma_topo if args.verify_graph else args.gamma) * A
                if val > bestval:
                    best, bestval = pk, val
            t_v = time.time() - t2
            t3 = time.time()
            rr = refine(q, ref, best, cfg) if not args.no_refine else None
            torch.cuda.synchronize(); t_r = time.time() - t3
            pu, pv = (rr.u, rr.v) if rr else (best.u, best.v)
            if args.verify_graph and mgraph is not None and vers:
                vb = vers[res.peaks.index(best)]
                rec.update(topo_n_query=vb.n_query, topo_matched=vb.n_matched, topo_match_ratio=vb.match_ratio,
                           topo_consistency=vb.topo_consistency, topo_score=vb.score,
                           topo_score_top1=vers[0].score, topo_ekeland=vb.ekeland)
                if vb.n_matched >= 3:
                    rs = LG.ransac_sim2(qg.J[vb.pairs[:, 0]], mgraph.J[vb.pairs[:, 1]], thr=2.0)
                    if rs is not None:
                        ru, rv = rs["t"] / args.gsd
                        rec.update(err_ransac_m=math.hypot(ru - gu, rv - gv) * args.gsd, ransac_inliers=rs["inliers"],
                                   ransac_residual=rs["residual"])
                        if args.use_ransac_pose and rs["inliers"] >= args.min_ransac_inliers:
                            pu, pv = ru, rv
            cents = np.array([p.mean(0) for p in q.polygons]) if q.polygons else np.zeros((1, 2))
            foot = math.sqrt(q.valid.sum()) * args.gsd
            rec.update(
                pred_u=pu, pred_v=pv, err_m=math.hypot(pu - gu, pv - gv) * args.gsd,
                err_grid_m=math.hypot(best.u - gu, best.v - gv) * args.gsd,
                top1_grid_err_m=math.hypot(res.peaks[0].u - gu, res.peaks[0].v - gv) * args.gsd,
                theta=rr.theta_deg if rr else best.theta_deg, s=rr.s if rr else best.s,
                J1=res.peaks[0].score, J2=res.second_score, peak_ratio=res.peaks[0].score - res.second_score,
                A=Avals[res.peaks.index(best)], A_top1=Avals[0], reranked=int(best is not res.peaks[0]),
                sigma_pos_m=rr.sigma_pos_m if rr else float("nan"),
                log_sigma_pos=math.log(max(rr.sigma_pos_m, 1e-3)) if rr else float("nan"),
                spread=float(np.sqrt(np.trace(np.cov(cents.T))) / max(foot, 1)) if len(cents) > 2 else 0.0,
                peak_rank_gt=int(np.argmin([math.hypot(p.u - gu, p.v - gv) for p in res.peaks])),
                any_peak_within_25m=int(min(math.hypot(p.u - gu, p.v - gv) for p in res.peaks) * args.gsd < 25),
                status="ok", t_query=t_q, t_search=t_s, t_verify=t_v, t_refine=t_r,
            )
            rows.append(rec)
            if len(rows) % 10 == 0:
                _flush(rows, outcsv); rows = []
                print(f"[{site}] {qi+1}/{len(df)}  last err {rec['err_m']:.1f} m", flush=True)
        _flush(rows, outcsv)
        d = pd.read_csv(outcsv); d = d[d["site"].astype(str).str.zfill(2) == site]
        if len(d):
            print(f"[{site}] n={len(d)} median={d.err_m.median():.1f}m  S@25={np.mean(d.err_m<25):.3f}  "
                  f"S@50={np.mean(d.err_m<50):.3f}", flush=True)


def _flush(rows, outcsv):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(outcsv, mode="a", header=not outcsv.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["satcache", "uavcache", "calib", "run", "satgraph"])
    ap.add_argument("--root", default="data/UAV-VisLoc")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--sites", nargs="+", default=[f"{i:02d}" for i in range(1, 12)])
    ap.add_argument("--calib-sites", nargs="+", default=["05"])
    ap.add_argument("--calib-n", type=int, default=30)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gsd", type=float, default=0.6)
    ap.add_argument("--sigma-m", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--k0", type=float, default=1.8, help="initial FOV constant c = 2 tan(HFOV/2) (84 deg -> 1.80)")
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--radius", type=float, default=1000.0, help="prior window radius (m); <=0 = global")
    ap.add_argument("--prior-frac", type=float, default=0.5)
    ap.add_argument("--agl", default="dem", choices=["dem", "raw"], help="height above ground from a public DEM at the prior window")
    ap.add_argument("--theta-range", type=float, default=10.0)
    ap.add_argument("--theta-step", type=float, default=2.5)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.8, 0.87, 0.93, 1.0, 1.07, 1.15, 1.25])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--coupled", type=int, default=1)
    ap.add_argument("--no-edge", action="store_true")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-graph", action="store_true", help="line-graph topological verification + RANSAC Sim(2)")
    ap.add_argument("--gamma-topo", type=float, default=0.3)
    ap.add_argument("--isotropic", action="store_true", help="ablation: ignore line orientation (plain chamfer)")
    ap.add_argument("--no-ekeland", action="store_true", help="ablation: no Ekeland free-cone term in verification")
    ap.add_argument("--use-ransac-pose", action="store_true")
    ap.add_argument("--min-ransac-inliers", type=int, default=4)
    ap.add_argument("--frontend", default="maskrcnn", choices=["maskrcnn", "lines"],
                    help="lines = training-free LSD line structure (no learned component)")
    ap.add_argument("--oracle", action="store_true", help="query structure = satellite segmentation under the true footprint")
    ap.add_argument("--local-radius", type=float, default=0.0, help="sanity: search only this radius (m) around GT")
    ap.add_argument("--out", default="results/reg/run.csv")
    args = ap.parse_args()
    args.sites = [s.zfill(2) for s in args.sites]
    args.calib_sites = [s.zfill(2) for s in args.calib_sites]
    {"satcache": stage_satcache, "uavcache": stage_uavcache, "calib": stage_calib, "run": stage_run,
     "satgraph": stage_satgraph}[args.stage](args)


if __name__ == "__main__":
    main()
