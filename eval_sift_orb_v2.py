"""
eval_sift_orb_v2.py  —  SIFT / ORB baseline cho UAV-VisLoc
Standalone: không phụ thuộc import từ project.
Đặt file ở BẤT KỲ đâu và chạy được.

Cách chạy:
    python eval_sift_orb_v2.py \
        --data-root /home/ngoc/UAV_nonGPS/UAV_nonGPS_dataset \
        --site 03 \
        --img-name 03_0001.JPG \
        --method both
"""

# lệnh chạy: 

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import argparse
import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Geo helpers
# ─────────────────────────────────────────────────────────────────────────────
EARTH_R = 6_378_137.0

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(max(0.0, a)))

def latlon_to_pixel(lat, lon, bounds, sat_w, sat_h):
    lt_lat, lt_lon = bounds['LT_lat'], bounds['LT_lon']
    rb_lat, rb_lon = bounds['RB_lat'], bounds['RB_lon']
    x = int(round((lon - lt_lon) / (rb_lon - lt_lon) * (sat_w - 1)))
    y = int(round((lt_lat - lat) / (lt_lat - rb_lat) * (sat_h - 1)))
    return max(0, min(x, sat_w-1)), max(0, min(y, sat_h-1))

def pixel_to_latlon(px, py, bounds, sat_w, sat_h):
    lt_lat, lt_lon = bounds['LT_lat'], bounds['LT_lon']
    rb_lat, rb_lon = bounds['RB_lat'], bounds['RB_lon']
    lat = lt_lat - (py / max(sat_h-1,1)) * (lt_lat - rb_lat)
    lon = lt_lon + (px / max(sat_w-1,1)) * (rb_lon - lt_lon)
    return float(lat), float(lon)

# ─────────────────────────────────────────────────────────────────────────────
# Load satellite bounds  (đọc satellite_coordinates_range.csv)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_satellite_name(name):
    stem = os.path.splitext(os.path.basename(str(name)))[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _extract_numeric_id(name):
    norm = _normalize_satellite_name(name)
    matches = re.findall(r"\d+", norm)
    if not matches:
        return None
    return matches[-1].lstrip("0") or "0"


def load_bounds(data_root, site):
    csv_path = Path(data_root) / "satellite_coordinates_range.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    satellite_filename = f"satellite{site}.tif"
    map_col = df.columns[0]
    map_names = df[map_col].astype(str).fillna("")
    map_norm = map_names.map(_normalize_satellite_name)
    map_norm_no_prefix = map_norm.str.replace(r"^satellite", "", regex=True)

    target_norm = _normalize_satellite_name(satellite_filename)
    target_no_prefix = re.sub(r"^satellite", "", target_norm)
    target_numeric = _extract_numeric_id(satellite_filename)

    exact = (
        (map_norm == target_norm)
        | (map_norm == target_no_prefix)
        | (map_norm_no_prefix == target_norm)
        | (map_norm_no_prefix == target_no_prefix)
    )
    if exact.any():
        row = df.loc[exact].iloc[0]
        return {
            'LT_lat': float(row['LT_lat_map']),
            'LT_lon': float(row['LT_lon_map']),
            'RB_lat': float(row['RB_lat_map']),
            'RB_lon': float(row['RB_lon_map']),
        }

    if target_numeric is not None:
        numeric_match = map_names.map(_extract_numeric_id) == target_numeric
        if numeric_match.any():
            row = df.loc[numeric_match].iloc[0]
            return {
                'LT_lat': float(row['LT_lat_map']),
                'LT_lon': float(row['LT_lon_map']),
                'RB_lat': float(row['RB_lat_map']),
                'RB_lon': float(row['RB_lon_map']),
            }

    raise RuntimeError(f"Không tìm thấy bounds cho site {site} trong {csv_path}")


def sliding_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


# ─────────────────────────────────────────────────────────────────────────────
# Tile satellite map
# ─────────────────────────────────────────────────────────────────────────────
def tile_satellite(sat_shape, patch_size=500, stride=100):
    h, w = sat_shape[:2]
    patches = []
    for top in sliding_positions(h, patch_size, stride):
        for left in sliding_positions(w, patch_size, stride):
            cx = left + patch_size // 2
            cy = top  + patch_size // 2
            patches.append((left, top, cx, cy))
    return patches


def extract_patch(sat_bgr, left, top, patch_size):
    return sat_bgr[top:top+patch_size, left:left+patch_size]


def downscale_for_baseline(img, max_side=2000):
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img, 1.0
    scale = max_side / side
    resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


# ─────────────────────────────────────────────────────────────────────────────
# SIFT / ORB matching
# ─────────────────────────────────────────────────────────────────────────────


def _project_uav_center(H, uav_shape, patch_shape):
    uav_h, uav_w = uav_shape[:2]
    patch_h, patch_w = patch_shape[:2]
    center = np.array([[[0.5 * (uav_w - 1), 0.5 * (uav_h - 1)]]], dtype=np.float32)
    try:
        projected = cv2.perspectiveTransform(center, H)
    except cv2.error:
        return None
    px, py = projected.reshape(2)
    if not (np.isfinite(px) and np.isfinite(py)):
        return None
    if px < 0 or py < 0 or px > patch_w - 1 or py > patch_h - 1:
        return None
    return float(px), float(py)


def _pixel_to_latlon_from_match(global_x, global_y, bounds, sat_w, sat_h):
    pred_lat, pred_lon = pixel_to_latlon(global_x, global_y, bounds, sat_w, sat_h)
    return pred_lat, pred_lon


def _result_from_patch_center(cx, cy, bounds, sat_w, sat_h):
    pred_lat, pred_lon = pixel_to_latlon(cx, cy, bounds, sat_w, sat_h)
    return pred_lat, pred_lon, float(cx), float(cy)

# ─────────────────────────────────────────────────────────────────────────────
# Preprocess UAV image  (giống process_uav() trong dataset.py)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_uav(img_path, out_size=500):
    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"Không đọc được: {img_path}")
    h, w = img.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    crop = img[cy - side // 2:cy - side // 2 + side, cx - side // 2:cx - side // 2 + side]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


@dataclass
class SatellitePatch:
    left: int
    top: int
    cx: int
    cy: int
    kp_data: list
    des: np.ndarray | None

    def keypoints(self):
        return [cv2.KeyPoint(x=float(x), y=float(y), size=float(size), angle=float(angle),
                             response=float(response), octave=int(octave), class_id=int(class_id))
                for x, y, size, angle, response, octave, class_id in self.kp_data]

    @staticmethod
    def from_keypoints(left, top, cx, cy, kp, des):
        kp_data = [
            (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
            for k in kp
        ]
        return SatellitePatch(left=left, top=top, cx=cx, cy=cy, kp_data=kp_data, des=des)

    @property
    def kp(self):
        return self.keypoints()


@dataclass
class SatelliteReference:
    patches: list
    sat_w: int
    sat_h: int
    built_at: float


REFERENCE_DIR = Path(__file__).resolve().parent / ".cache" / "eval_sift_orb"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

_REFERENCE_MEM_CACHE = {}


def _reference_cache_key(data_root, site, method, patch_size, stride):
    return (str(Path(data_root).resolve()), str(site), str(method), int(patch_size), int(stride))


def _store_reference_cache(key, ref):
    _REFERENCE_MEM_CACHE[key] = ref


def _get_reference_cache(key):
    return _REFERENCE_MEM_CACHE.get(key)


def clear_reference_cache():
    _REFERENCE_MEM_CACHE.clear()


def reference_cache_path(data_root, site, method, patch_size=500, stride=100):
    key = f"{Path(data_root).resolve()}__{site}__{method}__ps{patch_size}__st{stride}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return REFERENCE_DIR / f"{safe}.pkl"


def build_or_load_reference(data_root, site, method, patch_size=500, stride=100):
    data_root = Path(data_root)
    key = _reference_cache_key(data_root, site, method, patch_size, stride)
    cached = _get_reference_cache(key)
    if cached is not None:
        print("  Loaded reference cache: memory")
        return cached

    cache_path = reference_cache_path(data_root, site, method, patch_size, stride)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                ref = pickle.load(f)
            _store_reference_cache(key, ref)
            print(f"  Loaded reference cache: {cache_path}")
            return ref
        except (EOFError, pickle.UnpicklingError, OSError, AttributeError, TypeError) as exc:
            try:
                cache_path.unlink()
            except OSError:
                pass
            print(f"  Cache invalid, rebuilding: {cache_path} ({exc})")

    site_dir = data_root / site
    sat_path = site_dir / f"satellite{site}.tif"

    print(f"  Building reference from {sat_path.name} ...")
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(str(sat_path)) as pil:
        orig_w, orig_h = pil.size
        scale = min(1.0, 2000 / max(orig_w, orig_h))
        if scale < 1.0:
            pil.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        sat_rgb = np.array(pil.convert("RGB"))
    sat_bgr = cv2.cvtColor(sat_rgb, cv2.COLOR_RGB2BGR)
    sat_h, sat_w = sat_bgr.shape[:2]
    print(f"  Satellite: {sat_w}×{sat_h} px (scaled x{scale:.4f})")

    t0 = time.time()
    patch_size = max(64, int(round(patch_size * scale)))
    stride = max(32, int(round(stride * scale)))
    if method == "sift":
        det = cv2.SIFT_create(nfeatures=500)
    else:
        det = cv2.ORB_create(nfeatures=500)

    ref_patches = []
    for left, top, cx, cy in tile_satellite(sat_bgr.shape, patch_size=patch_size, stride=stride):
        patch = extract_patch(sat_bgr, left, top, patch_size)
        kp, des = det.detectAndCompute(patch, None)
        ref_patches.append(SatellitePatch.from_keypoints(left=left, top=top, cx=cx, cy=cy, kp=kp, des=des))

    ref = SatelliteReference(
        patches=ref_patches,
        sat_w=sat_w,
        sat_h=sat_h,
        built_at=time.time(),
    )
    with cache_path.open("wb") as f:
        pickle.dump(ref, f, protocol=pickle.HIGHEST_PROTOCOL)
    _store_reference_cache(key, ref)
    print(f"  Built {len(ref_patches)} patches in {time.time()-t0:.1f}s")
    print(f"  Saved reference cache: {cache_path}")
    return ref

def make_detector(method):
    if method == 'sift':
        det     = cv2.SIFT_create(nfeatures=500)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        det     = cv2.ORB_create(nfeatures=500)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return det, matcher


def match_uav(uav_bgr, ref: SatelliteReference, method, matcher,
              bounds, lowe_ratio=0.75, min_inliers=4):
    uav_gray = cv2.cvtColor(uav_bgr, cv2.COLOR_BGR2GRAY)
    uav_gray = cv2.equalizeHist(uav_gray)
    uav_bgr = cv2.cvtColor(uav_gray, cv2.COLOR_GRAY2BGR)

    """Match UAV query against prebuilt satellite reference."""
    det, _ = make_detector(method)
    kp_uav, des_uav = det.detectAndCompute(uav_bgr, None)
    if des_uav is None or len(kp_uav) < 4:
        return None, None, 0, None, None

    best_inliers = 0
    best_match = None
    sat_w, sat_h = ref.sat_w, ref.sat_h

    for sat_patch in ref.patches:
        if sat_patch.des is None or len(sat_patch.kp_data) < 4:
            continue
        left, top, cx, cy = sat_patch.left, sat_patch.top, sat_patch.cx, sat_patch.cy
        kp_sat, des_sat = sat_patch.kp, sat_patch.des
        try:
            matches = matcher.knnMatch(des_uav, des_sat, k=2)
        except cv2.error:
            continue
        good = [m for pair in matches
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < lowe_ratio * n.distance]
        if len(good) < min_inliers:
            continue
        src = np.float32([kp_uav[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp_sat[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        try:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        except cv2.error:
            continue
        inliers = int(mask.sum()) if mask is not None else 0
        if H is None or inliers <= 0:
            continue
        if inliers > best_inliers:
            best_inliers = inliers
            best_match = (sat_patch, left, top, cx, cy, H)

    if best_match is None:
        return None, None, 0, None, None

    sat_patch, left, top, cx, cy, H = best_match
    projected = _project_uav_center(H, uav_bgr.shape, (500, 500, 3))
    if projected is None:
        pred_lat, pred_lon, global_x, global_y = _result_from_patch_center(cx, cy, bounds, sat_w, sat_h)
        return pred_lat, pred_lon, best_inliers, global_x, global_y

    local_x, local_y = projected
    global_x = min(max(left + local_x, 0.0), float(sat_w - 1))
    global_y = min(max(top + local_y, 0.0), float(sat_h - 1))
    pred_lat, pred_lon = _pixel_to_latlon_from_match(global_x, global_y, bounds, sat_w, sat_h)
    return pred_lat, pred_lon, best_inliers, global_x, global_y

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(data_root, site, method, img_name=None, output_csv=None):
    data_root = Path(data_root)
    site_dir = data_root / site
    drone_dir = site_dir / "drone"
    csv_path = site_dir / f"{site}.csv"

    df = pd.read_csv(csv_path, sep=r"\s+|,", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df["filename"] = df["filename"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    if img_name:
        df = df[df["filename"] == img_name]
        if df.empty:
            raise ValueError(f"Không tìm thấy '{img_name}' trong {csv_path}")

    bounds = load_bounds(data_root, site)
    print(f"Bounds: {bounds}")

    ref_status = "memory" if _get_reference_cache(_reference_cache_key(data_root, site, method, 500, 100)) is not None else "disk/new"
    print(f"[{method.upper()}] Reference status for site {site}: {ref_status}")
    t_offline = time.time()
    ref = build_or_load_reference(data_root, site, method)
    offline_elapsed = time.time() - t_offline
    print(f"  Offline time: {offline_elapsed:.1f}s")

    _, matcher = make_detector(method)

    results = []
    total = len(df)
    online_total = 0.0

    print(f"\n[{method.upper()}] Online phase: evaluating {total} UAV queries ...")
    for i, (_, row) in enumerate(df.iterrows()):
        fname = row["filename"]
        gt_lat = float(row["lat"])
        gt_lon = float(row["lon"])

        img_path = drone_dir / fname
        if not img_path.exists():
            print(f"  [{i+1}/{total}] SKIP: {fname}")
            continue

        t_img = time.time()
        try:
            uav_bgr = preprocess_uav(img_path)
            pred_lat, pred_lon, inliers, pred_x, pred_y = match_uav(uav_bgr, ref, method, matcher, bounds)
        except Exception as e:
            print(f"  [{i+1}/{total}] error: {e}")
            continue

        elapsed = time.time() - t_img
        online_total += elapsed
        if pred_lat is None:
            error_m = float('inf')
            matched = False
        else:
            error_m = haversine_m(gt_lat, gt_lon, pred_lat, pred_lon)
            matched = True

        results.append({
            'site': site, 'method': method, 'filename': fname,
            'gt_lat': gt_lat, 'gt_lon': gt_lon,
            'pred_lat': pred_lat, 'pred_lon': pred_lon,
            'pred_x': pred_x, 'pred_y': pred_y,
            'error_m': error_m, 'inliers': inliers,
            'matched': matched, 'time_s': elapsed,
        })

        err_str = f"{error_m:.1f} m" if matched else "NO MATCH"
        pixel_str = f", px=({pred_x:.1f},{pred_y:.1f})" if matched and pred_x is not None and pred_y is not None else ""
        print(f"  [{i+1}/{total}] {fname} → {err_str}{pixel_str} (inliers={inliers}, t={elapsed:.1f}s)")

    errors = np.array([r['error_m'] for r in results if r['matched']])
    print(f"\n{'─'*50}")
    print(f"[{method.upper()}] Site {site} — {len(errors)}/{len(results)} matched")
    if len(errors) > 0:
        print(f"  Mean   : {np.mean(errors):.2f} m")
        print(f"  Median : {np.median(errors):.2f} m")
        print(f"  RMSE   : {np.sqrt(np.mean(errors**2)):.2f} m")
        print(f"  R@10m  : {np.mean(errors<=10)*100:.1f}%")
        print(f"  R@50m  : {np.mean(errors<=50)*100:.1f}%")
    print(f"  Online : {online_total:.1f}s")
    print(f"{'─'*50}")

    if output_csv and results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"  Saved → {output_csv}")
    return results

def evaluate_many(data_root, sites, method, image_names_by_site=None, output_csv=None):
    all_results = []
    data_root = Path(data_root)
    for site in sites:
        site_images = None if image_names_by_site is None else image_names_by_site.get(site)
        if site_images:
            for img_name in site_images:
                rows = evaluate(
                    data_root=data_root,
                    site=site,
                    method=method,
                    img_name=img_name,
                    output_csv=None,
                )
                all_results.extend(rows)
        else:
            rows = evaluate(
                data_root=data_root,
                site=site,
                method=method,
                img_name=None,
                output_csv=None,
            )
            all_results.extend(rows)

    if output_csv and all_results:
        pd.DataFrame(all_results).to_csv(output_csv, index=False)
        print(f"  Saved → {output_csv}")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--site", default=None, help="Single site, vd: 03")
    p.add_argument("--sites", default=None, help="Nhiều site, vd: 01,02,03")
    p.add_argument("--img-name", default=None,
                   help="Test 1 ảnh cho --site, vd: 03_0001.JPG")
    p.add_argument("--img-names", default=None,
                   help="Nhiều ảnh theo site, vd: 01:01_0001.JPG|01_0002.JPG,03:03_0001.JPG")
    p.add_argument("--method", choices=["sift", "orb", "both"], default="both")
    p.add_argument("--output-csv", default=None)
    return p.parse_args()


def _parse_sites(args):
    if args.sites:
        return [s.strip() for s in args.sites.split(",") if s.strip()]
    if args.site:
        return [args.site]
    return ["03"]


def _parse_image_names(args):
    if not args.img_names:
        return None
    mapping = {}
    for chunk in args.img_names.split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        site, names = chunk.split(":", 1)
        image_names = [name.strip() for name in names.split("|") if name.strip()]
        if image_names:
            mapping[site.strip()] = image_names
    return mapping or None


if __name__ == "__main__":
    args = parse_args()
    methods = ["sift", "orb"] if args.method == "both" else [args.method]
    sites = _parse_sites(args)
    image_names_by_site = _parse_image_names(args)

    for m in methods:
        if len(sites) == 1 and args.img_name:
            evaluate(
                data_root=args.data_root,
                site=sites[0],
                method=m,
                img_name=args.img_name,
                output_csv=args.output_csv,
            )
        else:
            rows = evaluate_many(
                data_root=args.data_root,
                sites=sites,
                method=m,
                image_names_by_site=image_names_by_site,
                output_csv=args.output_csv,
            )
            if rows:
                print(f"  Completed {m}: {len(rows)} rows")
