#!/usr/bin/env python
"""eval.py - one-command evaluation producing every OURS number and figure in the paper.

The paper (paper_draft/main.tex) reports, for the proposed method ("Ours"):
  * Main comparison rows: Mean / Median / RMSE / R@10m / R@50m / runtime,
    with and without compass-heading alignment (Table `tab:main`).
  * Per-site breakdown over the 11 UAV-VisLoc sites (Table `tab:persite`).
  * Season and altitude-band stratifications (Tables `tab:season`, `tab:altitude`).
  * Ablations: descriptor components, D_max sweep, ell_2 vs ell_1, K sweep,
    voting strategy (Table `tab:ablation`).
  * Sensitivity: IMU heading noise (+-5/10/15 deg), GSD error (+-10/20%).
  * Vote-distribution diagnostics and scalability (Table `tab:scalability`).

Stages (each resumable; per-site caching under --out):

  build    offline satellite descriptor DBs (GPU strongly recommended)
  query    online per-image localization -> records_<variant>.csv
  ablate   descriptor/matching ablations recomputed from caches (CPU only)
  report   metrics.json + tables.tex (paper-ready rows) + figures (CPU only)
  all      build -> query(main, no_heading) -> sensitivity subsets -> ablate -> report

Examples:

  python eval.py --estimate                          # resource estimate, runs nothing
  python eval.py --smoke                             # synthetic plumbing test (no GPU, no dataset)
  python eval.py --stage all                         # the full paper run
  python eval.py --stage build --sites 01 03         # build two sites only
  python eval.py --stage query --variant no_heading  # the "w/o heading" table row
  python eval.py --stage query --variant heading_noise --heading-noise 10
  python eval.py --stage query --variant gsd_error --gsd-error -0.2
  python eval.py --stage report                      # re-aggregate from cached records

Smoke testing on real data without committing to the full run:

  python eval.py --stage build --sites 01 --limit-patches 200
  python eval.py --stage query --sites 01 --limit-images 5 --examples 2

Nothing in this script changes the method: it drives the exact pipeline of
localization/ (paper hyperparameters by default) and aggregates the outputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Torch-free imports only at module level; torch-bound modules (segmentation,
# database.builder) are imported lazily inside the build/query stages.
from localization.database.kdtree import SatelliteDatabase
from localization.geometry.descriptor import triangle_descriptors_from_polygon
from localization.io.bounds import (
    latlon_to_pixel,
    load_satellite_bounds,
    pixel_offset_to_meters,
    pixel_to_latlon,
)
from localization.io.dataset import VisLocFlight, get_image_pose, load_flight_metadata
from localization.matching.query import query_uav

ALL_SITES = [f"{i:02d}" for i in range(1, 12)]
RECALL_THRESHOLDS_M = (10.0, 20.0, 50.0)
ALTITUDE_BANDS = ((400.0, 700.0), (700.0, 1200.0), (1200.0, 2000.0))

RECORD_FIELDS = [
    "site", "image", "variant",
    "n_polygons", "n_descriptors", "k", "total_votes",
    "pred_patch_id", "pred_px_x", "pred_px_y", "pred_lat", "pred_lon",
    "gt_lat", "gt_lon", "gt_px_x", "gt_px_y", "error_m",
    "votes", "second_votes", "margin",
    "gt_patch_id", "gt_patch_votes", "gt_patch_rank",
    "height", "date", "season", "alt_band",
    "t_preprocess", "t_segment", "t_descriptor", "t_query", "t_total",
]


@dataclass
class EvalConfig:
    """Paper §4.6 hyperparameters (Table tab:impl). Do not change for paper runs."""

    patch_size: int = 500
    stride: int = 100
    score_threshold: float = 0.5
    min_polygon_area: float = 50.0
    tolerance_px: float = 2.0
    max_depth: int = 4          # D_max
    leaf_size: int = 40
    k: int = 5                  # K nearest neighbours
    top_n: int = 100
    batch_size: int = 8
    sensitivity_limit: int = 100  # images/site for heading-noise & GSD sweeps
    ablation_k: Tuple[int, ...] = (1, 3, 10, 20)
    ablation_dmax: Tuple[int, ...] = (0, 1, 2, 8)
    seed: int = 42


# --------------------------------------------------------------------------- paths


class SitePaths:
    """All cached artifacts for one site under ``<out>/<site>/``."""

    def __init__(self, out_root: Path, site: str):
        self.site = site
        self.root = Path(out_root) / site
        self.desc_csv = self.root / "satellite_descriptors.csv"
        self.db_npz = self.root / "satellite_kdtree.npz"
        self.sat_polys = self.root / "satellite_polygons.jsonl.gz"
        self.meta_json = self.root / "site_meta.json"
        self.uav_polys_dir = self.root / "uav_polygons"
        self.uav_desc_dir = self.root / "uav_descriptors"
        self.figures_dir = self.root / "figures"

    def records_csv(self, variant: str = "main") -> Path:
        return self.root / f"records_{variant}.csv"

    def load_meta(self) -> dict:
        with open(self.meta_json, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_meta(self, meta: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def results_dir(out_root: Path) -> Path:
    d = Path(out_root) / "results"
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def detect_sites(data_root: Path) -> List[str]:
    """Sites whose satellite GeoTIFF is present on disk."""
    found = []
    for site in ALL_SITES:
        if (Path(data_root) / site / f"satellite{site}.tif").exists():
            found.append(site)
    return found


# --------------------------------------------------------------------------- metrics


def compute_metrics(errors: Sequence[float]) -> dict:
    """Mean / Median / RMSE / R@{10,20,50}m over finite errors (paper §6.1)."""
    e = np.asarray([x for x in errors if x is not None and np.isfinite(x)], dtype=np.float64)
    out = {"count": int(e.size)}
    if e.size == 0:
        return out
    out.update(
        mean_m=float(e.mean()),
        median_m=float(np.median(e)),
        rmse_m=float(np.sqrt((e ** 2).mean())),
    )
    for thr in RECALL_THRESHOLDS_M:
        out[f"recall_at_{int(thr)}m_pct"] = float(100.0 * (e <= thr).mean())
    return out


def season_of(date_str: str) -> str:
    """Map a capture date to the UAV-VisLoc season strata (summer/autumn)."""
    try:
        month = pd.to_datetime(str(date_str), errors="coerce").month
    except Exception:
        month = None
    if month is None or (isinstance(month, float) and math.isnan(month)):
        return "unknown"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "other"


def altitude_band(height: Optional[float]) -> str:
    if height is None or not np.isfinite(height):
        return "unknown"
    for lo, hi in ALTITUDE_BANDS:
        if lo <= height < hi:
            return f"{int(lo)}-{int(hi)}m"
    return "other"


def error_meters(pred_xy, gt_xy, bounds: dict, sat_w: int, sat_h: int) -> float:
    off = pixel_offset_to_meters(
        float(pred_xy[0]) - float(gt_xy[0]),
        float(pred_xy[1]) - float(gt_xy[1]),
        bounds, sat_w, sat_h,
    )
    return float(off["distance_m"])


# --------------------------------------------------------------------------- shared helpers


def polygons_to_descriptors(
    polygons, max_depth: int, offset_xy: Tuple[float, float] = (0.0, 0.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """Steps 3-4 over a polygon list (torch-free twin of builder._polygons_to_descriptors)."""
    descs, cents = [], []
    for poly in polygons:
        d, c = triangle_descriptors_from_polygon(poly, max_depth=int(max_depth))
        if d.shape[0] == 0:
            continue
        c = c.copy()
        c[:, 0] += float(offset_xy[0])
        c[:, 1] += float(offset_xy[1])
        descs.append(d)
        cents.append(c)
    if not descs:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.vstack(descs), np.vstack(cents)


def patch_centers_from_desc_csv(desc_csv: Path, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Geometric centers (tl + patch/2) of every unique patch in a descriptor CSV."""
    df = pd.read_csv(desc_csv, usecols=["patch_id", "top_left_x", "top_left_y"])
    grouped = df.drop_duplicates("patch_id")
    ids = grouped["patch_id"].to_numpy(dtype=object)
    centers = grouped[["top_left_x", "top_left_y"]].to_numpy(dtype=np.float64) + patch_size / 2.0
    return ids, centers


def gt_patch_diagnostics(result, gt_xy, patch_ids: np.ndarray, patch_centers: np.ndarray):
    """(gt_patch_id, votes on it, its vote rank) for the vote-distribution analysis."""
    if gt_xy is None or patch_ids.size == 0:
        return "", "", ""
    d2 = (patch_centers[:, 0] - gt_xy[0]) ** 2 + (patch_centers[:, 1] - gt_xy[1]) ** 2
    gt_pid = str(patch_ids[int(np.argmin(d2))])
    votes = int(result.all_votes.get(gt_pid, 0))
    if votes == 0:
        return gt_pid, 0, ""
    rank = 1 + sum(1 for v in result.all_votes.values() if v > votes)
    return gt_pid, votes, rank


def append_record(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RECORD_FIELDS})


def already_recorded(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        return set(pd.read_csv(csv_path, usecols=["image"])["image"].astype(str))
    except Exception:
        return set()


# --------------------------------------------------------------------------- stage: build


def stage_build(args, cfg: EvalConfig) -> None:
    import torch  # noqa: PLC0415

    from localization.database.builder import build_satellite_descriptors  # noqa: PLC0415
    from localization.segmentation.model import load_model  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    Image.MAX_IMAGE_PIXELS = None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[build] device = {device}")
    model = load_model(args.model, device=device, num_classes=2, pretrained=False).to(device).eval()

    for site in args.sites:
        sp = SitePaths(args.out, site)
        if sp.db_npz.exists() and not args.force:
            print(f"[build] {site}: DB exists, skipping (use --force to rebuild)")
            continue
        flight = VisLocFlight(flight_id=site, root=Path(args.data_root))
        if not flight.satellite_tif.exists():
            print(f"[build] {site}: satellite TIF missing, skipping")
            continue
        sp.root.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        sink_file = gzip.open(sp.sat_polys, "wt", encoding="utf-8")

        def sink(pid, tl, polys, _f=sink_file):
            _f.write(json.dumps({"patch_id": pid, "tl": [int(tl[0]), int(tl[1])], "polygons": polys}) + "\n")

        try:
            df = build_satellite_descriptors(
                tif_path=str(flight.satellite_tif),
                model=model,
                device=device,
                patch_size=cfg.patch_size,
                stride=cfg.stride,
                score_threshold=cfg.score_threshold,
                min_polygon_area=cfg.min_polygon_area,
                tolerance_px=cfg.tolerance_px,
                max_depth=cfg.max_depth,
                batch_size=cfg.batch_size,
                output_csv=str(sp.desc_csv),
                polygon_sink=sink,
                max_patches=args.limit_patches,
            )
        finally:
            sink_file.close()
        build_s = time.perf_counter() - t0

        db = SatelliteDatabase.from_dataframe(df, parent_tif=flight.satellite_tif.name, leaf_size=cfg.leaf_size)
        db.save(str(sp.db_npz))

        with Image.open(flight.satellite_tif) as im:
            sat_w, sat_h = im.size
        bounds = load_satellite_bounds(flight.satellite_tif.name, csv_path=str(flight.bounds_csv))
        sp.save_meta(
            {
                "site": site,
                "parent_tif": flight.satellite_tif.name,
                "sat_width": sat_w,
                "sat_height": sat_h,
                "bounds": bounds,
                "n_triangles": int(db.size),
                "n_patches_with_buildings": int(db.n_patches),
                "build_seconds": build_s,
                "limit_patches": args.limit_patches,
            }
        )
        print(f"[build] {site}: {db.size} triangles / {db.n_patches} patches in {build_s/60:.1f} min")


# --------------------------------------------------------------------------- stage: query


def resolve_variant(args) -> Tuple[str, dict]:
    """Variant name + per-image preprocessing kwargs."""
    if args.variant == "main":
        return "main", {}
    if args.variant == "no_heading":
        return "no_heading", {"apply_yaw": False}
    if args.variant == "heading_noise":
        mag = float(args.heading_noise)
        return f"heading_noise_{int(mag)}", {"heading_noise_mag": mag}
    if args.variant == "gsd_error":
        err = float(args.gsd_error)
        tag = f"{'+' if err >= 0 else '-'}{int(round(abs(err) * 100))}"
        return f"gsd_error_{tag}", {"scale_error": 1.0 + err}
    raise ValueError(f"unknown variant {args.variant!r}")


def stage_query(args, cfg: EvalConfig) -> None:
    import torch  # noqa: PLC0415

    from localization.preprocess.uav import process_uav  # noqa: PLC0415
    from localization.segmentation.inference import segment_image  # noqa: PLC0415
    from localization.segmentation.model import load_model  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    Image.MAX_IMAGE_PIXELS = None
    variant, vkwargs = resolve_variant(args)
    heading_noise_mag = vkwargs.pop("heading_noise_mag", None)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[query] variant={variant} device={device}")
    model = load_model(args.model, device=device, num_classes=2, pretrained=False).to(device).eval()

    limit = args.limit_images
    if limit is None and variant.startswith(("heading_noise", "gsd_error")):
        limit = cfg.sensitivity_limit  # sensitivity sweeps run on a fixed subset

    for site in args.sites:
        sp = SitePaths(args.out, site)
        if not sp.db_npz.exists():
            print(f"[query] {site}: no DB (run --stage build first), skipping")
            continue
        db = SatelliteDatabase.load(str(sp.db_npz))
        meta = sp.load_meta()
        bounds, sat_w, sat_h = meta.get("bounds"), meta["sat_width"], meta["sat_height"]
        patch_ids, patch_centers = patch_centers_from_desc_csv(sp.desc_csv, cfg.patch_size)

        flight = VisLocFlight(flight_id=site, root=Path(args.data_root))
        metadata_df = load_flight_metadata(flight.metadata_csv)
        images = flight.list_drone_images()
        if limit is not None:
            images = images[: int(limit)]

        rec_csv = sp.records_csv(variant)
        done = already_recorded(rec_csv)
        sp.uav_polys_dir.mkdir(parents=True, exist_ok=True)
        sp.uav_desc_dir.mkdir(parents=True, exist_ok=True)
        n_examples_left = int(args.examples or 0)

        print(f"[query] {site}: {len(images)} images ({len(done)} already done)")
        for img_idx, img_path in enumerate(images):
            name = img_path.name
            if name in done:
                continue
            kwargs = dict(vkwargs)
            if heading_noise_mag is not None:
                # Deterministic alternating sign so +-eps averages out per site.
                sign = 1.0 if (img_idx % 2 == 0) else -1.0
                kwargs["yaw_noise_deg"] = sign * heading_noise_mag

            t0 = time.perf_counter()
            try:
                _, img500, pmeta = process_uav(str(img_path), str(flight.metadata_csv), **kwargs)
            except Exception as exc:  # metadata row missing etc.
                print(f"  {name}: preprocess failed ({exc}); skipping")
                continue
            t_pre = time.perf_counter() - t0

            t1 = time.perf_counter()
            _mask, polygons = segment_image(
                image=img500, model=model, device=device,
                score_threshold=cfg.score_threshold,
                min_area=cfg.min_polygon_area,
                tolerance_px=cfg.tolerance_px,
            )
            t_seg = time.perf_counter() - t1

            t2 = time.perf_counter()
            desc, _cent = polygons_to_descriptors(polygons, max_depth=cfg.max_depth)
            t_desc = time.perf_counter() - t2

            if variant == "main":
                # Caches consumed by the ablation stage (CPU re-queries).
                with open(sp.uav_polys_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
                    json.dump({"image": name, "polygons": polygons}, f)
                np.save(sp.uav_desc_dir / f"{img_path.stem}.npy", desc)

            row = {
                "site": site, "image": name, "variant": variant,
                "n_polygons": len(polygons), "n_descriptors": int(desc.shape[0]),
                "k": cfg.k, "height": pmeta.get("height", ""),
                "t_preprocess": round(t_pre, 4), "t_segment": round(t_seg, 4),
                "t_descriptor": round(t_desc, 4),
            }
            pose = get_image_pose(metadata_df, name)
            gt_xy = None
            if pose is not None:
                row["date"] = pose.get("date", "")
                row["season"] = season_of(pose.get("date", ""))
                row["alt_band"] = altitude_band(float(pose["height"]))
                if bounds is not None:
                    gt_lat, gt_lon = float(pose["lat"]), float(pose["lon"])
                    gt_xy = latlon_to_pixel(gt_lat, gt_lon, bounds, sat_w, sat_h)
                    row.update(gt_lat=gt_lat, gt_lon=gt_lon, gt_px_x=gt_xy[0], gt_px_y=gt_xy[1])

            if desc.shape[0] == 0:
                row.update(t_query=0.0, t_total=round(t_pre + t_seg + t_desc, 4), total_votes=0)
                append_record(rec_csv, row)
                continue

            t3 = time.perf_counter()
            result = query_uav(desc, db, k=cfg.k, top_n=cfg.top_n)
            t_query = time.perf_counter() - t3
            row.update(t_query=round(t_query, 4), t_total=round(t_pre + t_seg + t_desc + t_query, 4))
            if result is None:
                append_record(rec_csv, row)
                continue

            pred_lat, pred_lon = ("", "")
            if bounds is not None:
                pred_lat, pred_lon = pixel_to_latlon(result.pixel_xy[0], result.pixel_xy[1], bounds, sat_w, sat_h)
            gt_pid, gt_votes, gt_rank = gt_patch_diagnostics(result, gt_xy, patch_ids, patch_centers)
            row.update(
                pred_patch_id=result.patch_id,
                pred_px_x=round(result.pixel_xy[0], 1), pred_px_y=round(result.pixel_xy[1], 1),
                pred_lat=pred_lat, pred_lon=pred_lon,
                votes=result.vote_count, second_votes=result.second_place_votes,
                margin=result.margin, total_votes=int(desc.shape[0]) * cfg.k,
                gt_patch_id=gt_pid, gt_patch_votes=gt_votes, gt_patch_rank=gt_rank,
            )
            if gt_xy is not None and bounds is not None:
                row["error_m"] = round(error_meters(result.pixel_xy, gt_xy, bounds, sat_w, sat_h), 2)
            append_record(rec_csv, row)

            if variant == "main" and n_examples_left > 0 and gt_xy is not None:
                n_examples_left -= 1
                _save_example_figures(sp, flight, img_path, result, gt_xy, row.get("error_m"), pmeta)

        print(f"[query] {site}: done -> {rec_csv}")


def _save_example_figures(sp: SitePaths, flight, img_path, result, gt_xy, error_m, pmeta) -> None:
    """Qualitative figures (paper Figs. 7-11 / Fig. 5(a) style) for a few queries."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from localization.matching.visualize import draw_gt_vs_topn_centroids, render_top_n_result

        sp.figures_dir.mkdir(parents=True, exist_ok=True)
        fig = render_top_n_result(
            uav_image_path=str(img_path),
            satellite_image_path=str(flight.satellite_tif),
            top_n=result.top_n,
            gt_pixel_xy=gt_xy,
            error_distance_m=error_m if error_m != "" else None,
            zoom_radius_px=1500,
            highlight_top_k=5,
            title=f"{img_path.name} -> {result.patch_id}",
            output_path=str(sp.figures_dir / f"match_top100_{img_path.stem}.png"),
        )
        plt.close(fig)
        fig = draw_gt_vs_topn_centroids(
            satellite_image_path=str(flight.satellite_tif),
            top_n=result.top_n,
            gt_pixel_xy=gt_xy,
            title=f"Site {sp.site}, {pmeta.get('height', 0):.0f} m",
            output_path=str(sp.figures_dir / f"gt_vs_top100_{img_path.stem}.png"),
        )
        plt.close(fig)
    except Exception as exc:  # figures are best-effort; never fail the run
        print(f"  example figure failed: {exc}")


# --------------------------------------------------------------------------- stage: ablate


def _load_site_caches(sp: SitePaths):
    """(sat_df_arrays, uav descriptor map, gt map, bounds, sat size) from the main run."""
    df = pd.read_csv(sp.desc_csv)
    sat_desc = df[["alpha1", "alpha2", "e1", "e2", "e3"]].to_numpy(dtype=np.float32)
    sat_cent = df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    sat_pids = df["patch_id"].to_numpy(dtype=object)

    rec = pd.read_csv(sp.records_csv("main"))
    rec = rec[(rec["n_descriptors"] > 0) & rec["gt_px_x"].notna() & rec["error_m"].notna()]
    gt_map: Dict[str, Tuple[float, float]] = {}
    uav_map: Dict[str, np.ndarray] = {}
    for _, r in rec.iterrows():
        stem = Path(str(r["image"])).stem
        npy = sp.uav_desc_dir / f"{stem}.npy"
        if not npy.exists():
            continue
        d = np.load(npy)
        if d.shape[0] == 0:
            continue
        uav_map[str(r["image"])] = d.astype(np.float32)
        gt_map[str(r["image"])] = (float(r["gt_px_x"]), float(r["gt_px_y"]))

    meta = sp.load_meta()
    return (sat_desc, sat_cent, sat_pids), uav_map, gt_map, meta["bounds"], meta["sat_width"], meta["sat_height"]


def _requery_errors(
    db: SatelliteDatabase,
    uav_map: Dict[str, np.ndarray],
    gt_map: Dict[str, Tuple[float, float]],
    bounds: dict,
    sat_w: int,
    sat_h: int,
    k: int,
    p: float = 1,
    weighted: bool = False,
    top1: bool = False,
    desc_cols: Optional[Sequence[int]] = None,
) -> List[float]:
    errors: List[float] = []
    for image, desc in uav_map.items():
        d = desc[:, list(desc_cols)] if desc_cols is not None else desc
        if d.shape[0] == 0:
            continue
        if top1:
            # "Top-1 match (no voting)": single globally-nearest triangle decides.
            dist, idx = db.query(d, k=1, p=p)
            j = int(idx[int(np.argmin(dist[:, 0])), 0])
            code = int(db.patch_id_codes[j])
            pred_xy = db.patch_centroids[code]
        else:
            res = query_uav(d, db, k=int(k), top_n=0, p=p, weighted=weighted)
            if res is None:
                continue
            pred_xy = res.pixel_xy
        errors.append(error_meters(pred_xy, gt_map[image], bounds, sat_w, sat_h))
    return errors


def _sat_arrays_from_polygon_cache(sp: SitePaths, max_depth: int):
    """Recompute the satellite descriptor arrays from the cached polygons (CPU)."""
    descs, cents, pids = [], [], []
    with gzip.open(sp.sat_polys, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            d, c = polygons_to_descriptors(
                rec["polygons"], max_depth=max_depth, offset_xy=(rec["tl"][0], rec["tl"][1])
            )
            if d.shape[0] == 0:
                continue
            descs.append(d)
            cents.append(c)
            pids.extend([rec["patch_id"]] * d.shape[0])
    if not descs:
        return (np.zeros((0, 5), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, object))
    return np.vstack(descs), np.vstack(cents), np.asarray(pids, dtype=object)


def _uav_map_from_polygon_cache(sp: SitePaths, uav_map_keys, max_depth: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for image in uav_map_keys:
        path = sp.uav_polys_dir / f"{Path(image).stem}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        d, _ = polygons_to_descriptors(rec["polygons"], max_depth=max_depth)
        if d.shape[0] > 0:
            out[image] = d
    return out


def stage_ablate(args, cfg: EvalConfig) -> None:
    rdir = results_dir(args.out)
    ablation: Dict[str, Dict[str, List[float]]] = {}

    def add(variant: str, site: str, errors: List[float]) -> None:
        ablation.setdefault(variant, {})[site] = errors
        m = compute_metrics(errors)
        print(f"[ablate] {variant:28s} {site}: n={m.get('count', 0)} "
              f"mean={m.get('mean_m', float('nan')):.1f}m")

    for site in args.sites:
        sp = SitePaths(args.out, site)
        if not (sp.desc_csv.exists() and sp.records_csv("main").exists()):
            print(f"[ablate] {site}: missing main caches, skipping")
            continue
        (sat_desc, sat_cent, sat_pids), uav_map, gt_map, bounds, sat_w, sat_h = _load_site_caches(sp)
        if not uav_map:
            print(f"[ablate] {site}: no cached UAV descriptors, skipping")
            continue
        db5 = SatelliteDatabase(sat_desc, sat_cent, sat_pids, leaf_size=cfg.leaf_size)
        q = dict(gt_map=gt_map, bounds=bounds, sat_w=sat_w, sat_h=sat_h)

        # Matching/voting ablations (reuse main descriptors).
        for k in cfg.ablation_k:
            add(f"K={k}", site, _requery_errors(db5, uav_map, k=k, **q))
        add("l2_distance", site, _requery_errors(db5, uav_map, k=cfg.k, p=2, **q))
        add("top1_no_voting", site, _requery_errors(db5, uav_map, k=cfg.k, top1=True, **q))
        add("distance_weighted", site, _requery_errors(db5, uav_map, k=cfg.k, weighted=True, **q))

        # Descriptor-component ablations (column slices of the 5-D descriptor).
        db_a = SatelliteDatabase(sat_desc[:, :2], sat_cent, sat_pids, leaf_size=cfg.leaf_size)
        add("interior_angles_only", site, _requery_errors(db_a, uav_map, k=cfg.k, desc_cols=(0, 1), **q))
        db_e = SatelliteDatabase(sat_desc[:, 2:], sat_cent, sat_pids, leaf_size=cfg.leaf_size)
        add("mfca_only", site, _requery_errors(db_e, uav_map, k=cfg.k, desc_cols=(2, 3, 4), **q))

        # D_max sweep: recompute descriptors from polygon caches (no GPU needed).
        if not args.skip_dmax:
            for dmax in cfg.ablation_dmax:
                t0 = time.perf_counter()
                sd, sc, pid = _sat_arrays_from_polygon_cache(sp, max_depth=dmax)
                um = _uav_map_from_polygon_cache(sp, uav_map.keys(), max_depth=dmax)
                if sd.shape[0] == 0 or not um:
                    continue
                db_d = SatelliteDatabase(sd, sc, pid, leaf_size=cfg.leaf_size)
                add(f"Dmax={dmax}", site, _requery_errors(db_d, um, k=cfg.k, **q))
                print(f"          (Dmax={dmax} recompute {time.perf_counter() - t0:.0f}s)")

    out_path = rdir / "ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {v: {"per_site": {s: compute_metrics(e) for s, e in sites.items()},
                 "overall": compute_metrics([x for e in sites.values() for x in e])}
             for v, sites in ablation.items()},
            f, indent=2,
        )
    print(f"[ablate] wrote {out_path}")


# --------------------------------------------------------------------------- stage: report


def _collect_records(out_root: Path, sites: Sequence[str], variant: str) -> pd.DataFrame:
    frames = []
    for site in sites:
        p = SitePaths(out_root, site).records_csv(variant)
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"site": str}))
    if not frames:
        return pd.DataFrame(columns=RECORD_FIELDS)
    return pd.concat(frames, ignore_index=True)


def _variant_names(out_root: Path, sites: Sequence[str]) -> List[str]:
    names = set()
    for site in sites:
        for p in SitePaths(out_root, site).root.glob("records_*.csv"):
            names.add(p.stem.replace("records_", "", 1))
    return sorted(names)


def stage_report(args, cfg: EvalConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rdir = results_dir(args.out)
    fdir = rdir / "figures"
    report: dict = {"generated_by": "eval.py", "config": cfg.__dict__ | {"sites": list(args.sites)}}

    main = _collect_records(args.out, args.sites, "main")
    if main.empty:
        print("[report] no main records found; run --stage query first")
        return
    ok = main[main["error_m"].notna() & (main["n_descriptors"] > 0)].copy()
    report["coverage"] = {
        "queries_total": int(len(main)),
        "queries_with_descriptors": int((main["n_descriptors"] > 0).sum()),
        "coverage_pct": float(100.0 * (main["n_descriptors"] > 0).mean()),
        "avg_descriptors_per_query": float(main["n_descriptors"].mean()),
    }
    report["main"] = compute_metrics(ok["error_m"]) | {
        "runtime_total_s_mean": float(ok["t_total"].mean()),
        "runtime_query_s_mean": float(ok["t_query"].mean()),
        "runtime_segment_s_mean": float(ok["t_segment"].mean()),
    }

    report["per_site"] = {
        site: compute_metrics(g["error_m"]) | {"avg_M": float(g["n_descriptors"].mean())}
        for site, g in ok.groupby("site")
    }
    report["season"] = {s: compute_metrics(g["error_m"]) for s, g in ok.groupby("season")}
    report["altitude"] = {
        b: compute_metrics(g["error_m"]) | {"avg_M": float(g["n_descriptors"].mean())}
        for b, g in ok.groupby("alt_band")
    }
    report["vote_diagnostics_per_site"] = {
        site: {
            "inlier_vote_ratio_pct": float(100.0 * (g["gt_patch_votes"].fillna(0).astype(float)
                                                    / g["total_votes"].clip(lower=1)).mean()),
            "mean_margin": float(g["margin"].mean()),
            "min_winning_votes": int(g["votes"].min()) if len(g) else 0,
            "gt_in_top100_pct": float(100.0 * (pd.to_numeric(g["gt_patch_rank"], errors="coerce") <= 100).mean()),
        }
        for site, g in ok.groupby("site")
    }

    # Other measured variants (no_heading, heading_noise_*, gsd_error_*).
    report["variants"] = {}
    for v in _variant_names(args.out, args.sites):
        if v == "main":
            continue
        dfv = _collect_records(args.out, args.sites, v)
        okv = dfv[dfv["error_m"].notna()]
        report["variants"][v] = compute_metrics(okv["error_m"]) | {
            "runtime_total_s_mean": float(okv["t_total"].mean()) if len(okv) else None,
        }

    abl_path = rdir / "ablation.json"
    if abl_path.exists():
        with open(abl_path, "r", encoding="utf-8") as f:
            report["ablation"] = json.load(f)

    report["scalability"] = _scalability(args, cfg)

    with open(rdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_tables_tex(rdir / "tables.tex", report)
    _write_figures(fdir, ok, report, plt)
    print(f"[report] wrote {rdir / 'metrics.json'}, {rdir / 'tables.tex'}, figures -> {fdir}")


def _scalability(args, cfg: EvalConfig) -> List[dict]:
    """Memory + query latency vs number of simultaneously indexed sites (Table tab:scalability)."""
    site_arrays = []
    for site in args.sites:
        sp = SitePaths(args.out, site)
        if not sp.desc_csv.exists():
            continue
        df = pd.read_csv(sp.desc_csv)
        site_arrays.append(
            (site,
             df[["alpha1", "alpha2", "e1", "e2", "e3"]].to_numpy(dtype=np.float32),
             df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32),
             (site + "/" + df["patch_id"].astype(str)).to_numpy(dtype=object))
        )
    if not site_arrays:
        return []

    # A small pool of cached UAV queries to time against.
    uav_pool: List[np.ndarray] = []
    for site in args.sites:
        for npy in sorted(SitePaths(args.out, site).uav_desc_dir.glob("*.npy"))[:5]:
            d = np.load(npy)
            if d.shape[0] > 0:
                uav_pool.append(d.astype(np.float32))
        if len(uav_pool) >= 20:
            break

    out = []
    for n_sites in sorted({1, min(3, len(site_arrays)), len(site_arrays)}):
        subset = site_arrays[:n_sites]
        desc = np.vstack([a[1] for a in subset])
        cent = np.vstack([a[2] for a in subset])
        pids = np.concatenate([a[3] for a in subset])
        t0 = time.perf_counter()
        db = SatelliteDatabase(desc, cent, pids, leaf_size=cfg.leaf_size)
        _ = db.tree
        build_s = time.perf_counter() - t0
        q_times = []
        for d in uav_pool:
            t1 = time.perf_counter()
            query_uav(d, db, k=cfg.k, top_n=0)
            q_times.append(time.perf_counter() - t1)
        out.append(
            {
                "n_sites": n_sites,
                "n_triangles": int(desc.shape[0]),
                "memory_mb": round((desc.nbytes + cent.nbytes + db.patch_id_codes.nbytes) / 2 ** 20, 1),
                "tree_build_s": round(build_s, 2),
                "query_s_mean": round(float(np.mean(q_times)), 4) if q_times else None,
            }
        )
    return out


def _fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    return f"{x:.{nd}f}"


def _ablation_label(variant: str) -> str:
    """LaTeX-safe row labels matching the paper's Table tab:ablation wording."""
    fixed = {
        "l2_distance": r"Full 5-D CDT, $\ell_2$ distance",
        "top1_no_voting": r"Top-1 match (no voting)",
        "distance_weighted": r"Distance-weighted voting",
        "interior_angles_only": r"Interior angles only ($\alpha_1,\alpha_2$)",
        "mfca_only": r"MFCA only ($e_1,e_2,e_3$)",
    }
    if variant in fixed:
        return fixed[variant]
    if variant.startswith("K="):
        return f"$K={variant.split('=')[1]}$"
    if variant.startswith("Dmax="):
        return rf"Full 5-D CDT, $D_{{\max}}={variant.split('=')[1]}$"
    return variant.replace("_", r"\_")


def _write_tables_tex(path: Path, rep: dict) -> None:
    """Paper-ready LaTeX rows for every OURS table. Paste into paper_draft/main.tex."""
    L: List[str] = ["% Auto-generated by eval.py - OURS rows for paper_draft/main.tex", ""]
    m = rep.get("main", {})

    L.append("% --- Table tab:main (Ours rows) ---")
    L.append(
        f"\\textbf{{MFCA (w/ heading)}} & \\textbf{{MaskRCNN}} & \\textbf{{No}} & "
        f"\\textbf{{{_fmt(m.get('mean_m'))}}} & \\textbf{{{_fmt(m.get('median_m'))}}} & "
        f"\\textbf{{{_fmt(m.get('rmse_m'))}}} & \\textbf{{{_fmt(m.get('recall_at_10m_pct'))}}} & "
        f"\\textbf{{{_fmt(m.get('recall_at_50m_pct'))}}} & \\textbf{{{_fmt(m.get('runtime_total_s_mean'), 2)}}} \\\\"
    )
    nh = rep.get("variants", {}).get("no_heading", {})
    L.append(
        f"\\textbf{{MFCA (w/o heading)}} & \\textbf{{MaskRCNN}} & \\textbf{{No}} & "
        f"{_fmt(nh.get('mean_m'))} & {_fmt(nh.get('median_m'))} & {_fmt(nh.get('rmse_m'))} & "
        f"{_fmt(nh.get('recall_at_10m_pct'))} & {_fmt(nh.get('recall_at_50m_pct'))} & "
        f"{_fmt(nh.get('runtime_total_s_mean'), 2)} \\\\"
    )

    L.append("")
    L.append("% --- Table tab:persite (Ours / R@10m / R@50m columns) ---")
    for site, sm in sorted(rep.get("per_site", {}).items()):
        L.append(f"% site {site}: Ours={_fmt(sm.get('mean_m'))} & "
                 f"R@10m={_fmt(sm.get('recall_at_10m_pct'))} & R@50m={_fmt(sm.get('recall_at_50m_pct'))} "
                 f"(n={sm.get('count')}, avg M={_fmt(sm.get('avg_M'))})")

    L.append("")
    L.append("% --- Table tab:season ---")
    for season, sm in sorted(rep.get("season", {}).items()):
        L.append(f"% {season}: mean={_fmt(sm.get('mean_m'))} m (n={sm.get('count')})")

    L.append("")
    L.append("% --- Table tab:altitude ---")
    for band, sm in sorted(rep.get("altitude", {}).items()):
        L.append(f"{band.replace('-', '--')} & {_fmt(sm.get('mean_m'))} & "
                 f"{_fmt(sm.get('recall_at_20m_pct'))} & {_fmt(sm.get('avg_M'), 0)} \\\\"
                 f"  % n={sm.get('count')}")

    abl = rep.get("ablation", {})
    if abl:
        L.append("")
        L.append("% --- Table tab:ablation (Mean / R@20m) ---")
        for variant, vm in abl.items():
            o = vm.get("overall", {})
            L.append(f"{_ablation_label(variant)} & {_fmt(o.get('mean_m'))} & "
                     f"{_fmt(o.get('recall_at_20m_pct'))} \\\\")
        L.append(rf"\textbf{{Full 5-D CDT, $\ell_1$, $D_{{\max}}=4$, $K=5$}} & "
                 rf"\textbf{{{_fmt(m.get('mean_m'))}}} & "
                 rf"\textbf{{{_fmt(m.get('recall_at_20m_pct'))}}} \\  % from main run")

    sens = {k: v for k, v in rep.get("variants", {}).items() if k != "no_heading"}
    if sens:
        L.append("")
        L.append("% --- Sensitivity (heading noise / GSD error subsets) ---")
        for variant, vm in sorted(sens.items()):
            L.append(f"% {variant}: mean={_fmt(vm.get('mean_m'))} m, "
                     f"R@50m={_fmt(vm.get('recall_at_50m_pct'))}% (n={vm.get('count')})")

    scal = rep.get("scalability", [])
    if scal:
        L.append("")
        L.append("% --- Table tab:scalability ---")
        for r in scal:
            L.append(f"{r['n_sites']} & ${r['n_triangles'] / 1e6:.2f}$M & "
                     f"{r['memory_mb']} & {_fmt(r['query_s_mean'], 3)} \\\\")

    vd = rep.get("vote_diagnostics_per_site", {})
    if vd:
        L.append("")
        L.append("% --- Vote-distribution diagnostics (per site) ---")
        for site, d in sorted(vd.items()):
            L.append(f"% {site}: inlier-vote {_fmt(d['inlier_vote_ratio_pct'])}%, "
                     f"margin {_fmt(d['mean_margin'])}, min-win {d['min_winning_votes']}, "
                     f"GT-in-top100 {_fmt(d['gt_in_top100_pct'])}%")

    cov = rep.get("coverage", {})
    L.append("")
    L.append(f"% Coverage: {cov.get('queries_with_descriptors')}/{cov.get('queries_total')} queries "
             f"with >=1 descriptor ({_fmt(cov.get('coverage_pct'))}%); "
             f"errors computed on covered queries only.")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _write_figures(fdir: Path, ok: pd.DataFrame, rep: dict, plt) -> None:
    fdir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(fdir / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(fdir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 1) Error CDF, overall + per site.
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for site, g in sorted(ok.groupby("site")):
        e = np.sort(g["error_m"].to_numpy(dtype=float))
        if e.size:
            ax.plot(e, np.arange(1, e.size + 1) / e.size, lw=0.9, alpha=0.55, label=f"site {site}")
    e = np.sort(ok["error_m"].to_numpy(dtype=float))
    if e.size:
        ax.plot(e, np.arange(1, e.size + 1) / e.size, "k-", lw=2.2, label="all sites")
    ax.set_xscale("log")
    ax.set_xlabel("Localization error (m)")
    ax.set_ylabel("Fraction of queries")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    save(fig, "fig_error_cdf")

    # 2) Per-site mean/median bars.
    ps = rep.get("per_site", {})
    if ps:
        sites = sorted(ps)
        x = np.arange(len(sites))
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        ax.bar(x - 0.2, [ps[s].get("mean_m", np.nan) for s in sites], 0.4, label="Mean")
        ax.bar(x + 0.2, [ps[s].get("median_m", np.nan) for s in sites], 0.4, label="Median")
        ax.set_xticks(x, sites)
        ax.set_xlabel("UAV-VisLoc site")
        ax.set_ylabel("Error (m)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        save(fig, "fig_per_site_error")

    # 3-4) K and D_max sensitivity curves from the ablation results.
    abl = rep.get("ablation", {})
    for prefix, fname, xlabel in (("K=", "fig_k_sensitivity", "K nearest neighbours"),
                                  ("Dmax=", "fig_dmax_sensitivity", r"Kernel expansion depth $D_{\max}$")):
        pts = sorted(
            (int(v.split("=")[1]), vm["overall"].get("mean_m"))
            for v, vm in abl.items() if v.startswith(prefix) and vm.get("overall", {}).get("mean_m") is not None
        )
        if prefix == "K=" and rep.get("main", {}).get("mean_m") is not None:
            pts = sorted(pts + [(5, rep["main"]["mean_m"])])
        if prefix == "Dmax=" and rep.get("main", {}).get("mean_m") is not None:
            pts = sorted(pts + [(4, rep["main"]["mean_m"])])
        if len(pts) >= 2:
            fig, ax = plt.subplots(figsize=(4.2, 3.0))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean error (m)")
            ax.grid(alpha=0.3)
            save(fig, fname)

    # 5) Vote margin vs error (plurality-vote validation).
    if {"margin", "error_m"}.issubset(ok.columns) and len(ok):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.scatter(ok["margin"], ok["error_m"], s=6, alpha=0.35)
        ax.set_xlabel("Vote margin (winner - runner-up)")
        ax.set_ylabel("Error (m)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        save(fig, "fig_vote_margin_vs_error")

    # 6) Runtime breakdown.
    cols = ["t_preprocess", "t_segment", "t_descriptor", "t_query"]
    if set(cols).issubset(ok.columns) and len(ok):
        means = [float(ok[c].mean()) for c in cols]
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.bar(["Preprocess", "Segment", "Descriptor", "KD-query"], means)
        ax.set_ylabel("Mean time per query (s)")
        ax.grid(axis="y", alpha=0.3)
        save(fig, "fig_runtime_breakdown")


# --------------------------------------------------------------------------- estimate


def cmd_estimate(args, cfg: EvalConfig) -> None:
    """Resource estimate for the full run; reads the dataset when present."""
    from localization.database.patches import patch_count  # torch-free

    data_root = Path(args.data_root)
    rows, total_patches, total_images, have_data = [], 0, 0, False
    for site in ALL_SITES:
        tif = data_root / site / f"satellite{site}.tif"
        n_imgs = len(list((data_root / site / "drone").glob("*"))) if (data_root / site / "drone").exists() else 0
        n_patch = None
        if tif.exists():
            have_data = True
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(tif) as im:
                n_patch = patch_count(im.size, cfg.patch_size, cfg.stride)
        rows.append((site, n_patch, n_imgs))
        total_patches += n_patch or 0
        total_images += n_imgs

    if not have_data:
        # Dataset not on disk: use canonical UAV-VisLoc numbers.
        # satellite01 is 9774x26762 -> ~24.7k patches; maps vary, ~20k average.
        total_patches = 11 * 20_000
        total_images = 6_742
        print("Dataset not found on disk - using canonical UAV-VisLoc statistics.")
    else:
        print(f"{'site':>4} {'patches':>9} {'images':>7}")
        for site, n_patch, n_imgs in rows:
            print(f"{site:>4} {n_patch if n_patch is not None else '---':>9} {n_imgs:>7}")

    n_sens_imgs = min(cfg.sensitivity_limit, 999) * 11 * 7  # 3 heading + 4 GSD runs
    print(f"""
=== Resource estimate for `python eval.py --stage all` ===

Workload
  Satellite DB build : ~{total_patches:,} Mask R-CNN forward passes (500x500), once
  Main query run     : {total_images:,} images (preprocess + 1 segmentation + KD-query)
  w/o-heading run    : {total_images:,} images (same cost as main)
  Sensitivity sweeps : ~{n_sens_imgs:,} images ({cfg.sensitivity_limit}/site x 7 configs)
  Ablation stage     : CPU only; D_max sweep re-triangulates the cached polygons
                       of ~{total_patches:,} patches x {len(cfg.ablation_dmax)} depths (no GPU)

Estimated wall-clock (Mask R-CNN R50-FPN @500px, batch {cfg.batch_size})
                      build        query(2x)     sensitivity   ablate(CPU)   total
  A100 / 4090         ~1.5-2 h     ~2-3 h        ~1 h          ~3-6 h        ~8-12 h
  T4 / RTX 3060       ~4-6 h       ~4-6 h        ~1.5-2 h      ~3-6 h        ~13-20 h
  CPU only            ~80-120 h    ~30-50 h      not advised   ~3-6 h        infeasible

Memory / disk
  GPU VRAM            ~5-7 GB (batch {cfg.batch_size}; reduce --batch-size if OOM)
  System RAM          <= 4 GB (largest KD-tree ~6.6M x 5 float32 ~ 130 MB + overhead)
  Disk: dataset       ~17.7 GB (UAV-VisLoc)
  Disk: caches        ~1-3 GB  (descriptor CSVs ~12 MB/site, polygon caches
                       ~50-200 MB/site, per-image descriptor .npy files)
  Disk: results       < 100 MB (records, metrics.json, tables.tex, figures)

All stages are resumable: re-running skips completed sites/images, so the run
can be split across several GPU sessions (e.g. Colab/Kaggle).
""")


# --------------------------------------------------------------------------- smoke


def run_smoke(args, cfg: EvalConfig) -> int:
    """End-to-end plumbing test on synthetic geometry - no torch, no dataset.

    Builds a synthetic 'satellite map' of procedurally generated building
    footprints, runs the real descriptor/KD-tree/vote pipeline, then exercises
    the ablate and report stages over the same cache layout the real run uses.
    """
    print("=== eval.py smoke test (synthetic, torch-free) ===")
    out = Path(args.out) if args.out != "outputs/eval" else REPO_ROOT / "outputs" / "eval_smoke"
    rng = np.random.default_rng(cfg.seed)
    site = "01"
    sp = SitePaths(out, site)
    for d in (sp.uav_polys_dir, sp.uav_desc_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Clear previous smoke records so the test is deterministic.
    for p in sp.root.glob("records_*.csv"):
        p.unlink()

    map_w = map_h = 2000
    patch, stride = cfg.patch_size, 250  # coarser stride to keep the smoke DB small

    def make_polygon(cx, cy):
        # Star-convex polygon with random radii/angles: distinctive interior +
        # Ekeland angles, like real segmented footprints (vs. ambiguous rects).
        n = int(rng.integers(5, 10))
        angles = np.sort(rng.uniform(0, 2 * math.pi, size=n))
        radii = rng.uniform(15, 50, size=n)
        return [[cx + r * math.cos(a), cy + r * math.sin(a)] for a, r in zip(angles, radii)]

    # Synthetic 'build' stage: per-patch polygons -> descriptors + caches.
    xs = list(range(0, map_w - patch + 1, stride))
    ys = list(range(0, map_h - patch + 1, stride))
    records, polys_by_patch = [], {}
    with gzip.open(sp.sat_polys, "wt", encoding="utf-8") as f:
        for ri, y in enumerate(ys):
            for ci, x in enumerate(xs):
                pid = f"synthetic_{ri:04d}_{ci:04d}"
                polys = [make_polygon(rng.uniform(70, patch - 70), rng.uniform(70, patch - 70))
                         for _ in range(int(rng.integers(2, 6)))]
                polys_by_patch[pid] = (x, y, polys)
                f.write(json.dumps({"patch_id": pid, "tl": [x, y], "polygons": polys}) + "\n")
                d, c = polygons_to_descriptors(polys, max_depth=cfg.max_depth, offset_xy=(x, y))
                for dd, cc in zip(d, c):
                    records.append({"patch_id": pid, "parent_tif": "synthetic.tif",
                                    "top_left_x": x, "top_left_y": y,
                                    "centroid_x": float(cc[0]), "centroid_y": float(cc[1]),
                                    "alpha1": float(dd[0]), "alpha2": float(dd[1]),
                                    "e1": float(dd[2]), "e2": float(dd[3]), "e3": float(dd[4])})
    df = pd.DataFrame.from_records(records)
    sp.desc_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sp.desc_csv, index=False)
    db = SatelliteDatabase.from_dataframe(df, parent_tif="synthetic.tif", leaf_size=cfg.leaf_size)
    db.save(str(sp.db_npz))
    # ~0.3 m/px synthetic georeferencing.
    bounds = {"LT_lat": 30.0, "LT_lon": 114.0,
              "RB_lat": 30.0 - map_h * 0.3 / 111_320.0,
              "RB_lon": 114.0 + map_w * 0.3 / (111_320.0 * math.cos(math.radians(30.0)))}
    sp.save_meta({"site": site, "parent_tif": "synthetic.tif", "sat_width": map_w,
                  "sat_height": map_h, "bounds": bounds, "n_triangles": int(db.size),
                  "n_patches_with_buildings": int(db.n_patches), "build_seconds": 0.0})
    print(f"[smoke] synthetic DB: {db.size} triangles / {db.n_patches} patches")

    # Synthetic 'query' stage: jittered copies of a patch's polygons as queries.
    patch_ids, patch_centers = patch_centers_from_desc_csv(sp.desc_csv, patch)
    pids = list(polys_by_patch)
    n_hit = 0
    seasons = ["2023-07-15", "2023-10-02"]
    for qi in range(10):
        pid = pids[int(rng.integers(0, len(pids)))]
        x, y, polys = polys_by_patch[pid]
        jittered = [[[px + rng.normal(0, 0.8), py + rng.normal(0, 0.8)] for px, py in poly] for poly in polys]
        name = f"synthetic_q{qi:02d}.JPG"
        t0 = time.perf_counter()
        desc, _ = polygons_to_descriptors(jittered, max_depth=cfg.max_depth)
        t_desc = time.perf_counter() - t0
        with open(sp.uav_polys_dir / f"{Path(name).stem}.json", "w", encoding="utf-8") as f:
            json.dump({"image": name, "polygons": jittered}, f)
        np.save(sp.uav_desc_dir / f"{Path(name).stem}.npy", desc)
        t1 = time.perf_counter()
        result = query_uav(desc, db, k=cfg.k, top_n=cfg.top_n)
        t_query = time.perf_counter() - t1
        gt_xy = (x + patch / 2.0, y + patch / 2.0)
        err = error_meters(result.pixel_xy, gt_xy, bounds, map_w, map_h)
        if result.patch_id == pid:
            n_hit += 1
        gt_pid, gt_votes, gt_rank = gt_patch_diagnostics(result, gt_xy, patch_ids, patch_centers)
        gt_lat, gt_lon = pixel_to_latlon(gt_xy[0], gt_xy[1], bounds, map_w, map_h)
        height = float(rng.choice([450, 800, 1500]))
        append_record(sp.records_csv("main"), {
            "site": site, "image": name, "variant": "main",
            "n_polygons": len(jittered), "n_descriptors": int(desc.shape[0]), "k": cfg.k,
            "total_votes": int(desc.shape[0]) * cfg.k,
            "pred_patch_id": result.patch_id,
            "pred_px_x": round(result.pixel_xy[0], 1), "pred_px_y": round(result.pixel_xy[1], 1),
            "gt_lat": gt_lat, "gt_lon": gt_lon, "gt_px_x": gt_xy[0], "gt_px_y": gt_xy[1],
            "error_m": round(err, 2), "votes": result.vote_count,
            "second_votes": result.second_place_votes, "margin": result.margin,
            "gt_patch_id": gt_pid, "gt_patch_votes": gt_votes, "gt_patch_rank": gt_rank,
            "height": height, "date": seasons[qi % 2], "season": season_of(seasons[qi % 2]),
            "alt_band": altitude_band(height),
            "t_preprocess": 0.0, "t_segment": 0.0,
            "t_descriptor": round(t_desc, 4), "t_query": round(t_query, 4),
            "t_total": round(t_desc + t_query, 4),
        })
    # A synthetic 'no_heading' variant so the report exercises the variants path.
    main_df = pd.read_csv(sp.records_csv("main"))
    nh = main_df.copy()
    nh["variant"] = "no_heading"
    nh["error_m"] = nh["error_m"] * 3.0  # degraded, as expected without yaw alignment
    nh.to_csv(sp.records_csv("no_heading"), index=False)
    print(f"[smoke] 10 synthetic queries: rank-1 patch correct for {n_hit}/10")

    # Real ablate + report stages over the synthetic caches.
    smoke_args = argparse.Namespace(out=out, sites=[site], skip_dmax=False)
    smoke_cfg = EvalConfig(ablation_k=(1, 3), ablation_dmax=(1, 8))
    stage_ablate(smoke_args, smoke_cfg)
    stage_report(smoke_args, smoke_cfg)

    # Assertions.
    rdir = results_dir(out)
    failures = []
    if n_hit < 7:
        failures.append(f"rank-1 accuracy too low on synthetic data ({n_hit}/10)")
    with open(rdir / "metrics.json", "r", encoding="utf-8") as f:
        rep = json.load(f)
    if rep.get("main", {}).get("count", 0) < 8:
        failures.append("metrics.json missing main metrics")
    if "ablation" not in rep or not rep["ablation"]:
        failures.append("ablation results missing")
    if not (rdir / "tables.tex").exists():
        failures.append("tables.tex missing")
    n_figs = len(list((rdir / "figures").glob("*.png")))
    if n_figs < 3:
        failures.append(f"expected >=3 figures, found {n_figs}")
    med = rep.get("main", {}).get("median_m")
    if med is None or med > 100:
        failures.append(f"synthetic median error too high ({med})")

    print()
    if failures:
        for msg in failures:
            print(f"[FAIL] {msg}")
        return 1
    print(f"[PASS] smoke test OK - median synthetic error {med:.1f} m, "
          f"{n_figs} figures, outputs in {out}")
    return 0


# --------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["build", "query", "ablate", "report", "all"], default=None)
    p.add_argument("--data-root", default=str(REPO_ROOT / "UAV-VisLoc"))
    p.add_argument("--model", default=str(REPO_ROOT / "best_model.pth"))
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--sites", nargs="*", default=None, help="default: all sites with a satellite TIF on disk")
    p.add_argument("--variant", choices=["main", "no_heading", "heading_noise", "gsd_error"], default="main")
    p.add_argument("--heading-noise", type=float, default=5.0, help="degrees, for --variant heading_noise")
    p.add_argument("--gsd-error", type=float, default=0.1, help="signed fraction, for --variant gsd_error")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--limit-images", type=int, default=None, help="cap query images per site (testing)")
    p.add_argument("--limit-patches", type=int, default=None, help="cap build patches per site (testing)")
    p.add_argument("--examples", type=int, default=3, help="qualitative figures per site (main variant)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skip-dmax", action="store_true", help="skip the (slow) D_max ablation recompute")
    p.add_argument("--force", action="store_true", help="rebuild satellite DBs even if cached")
    p.add_argument("--smoke", action="store_true", help="synthetic plumbing test (no GPU/dataset)")
    p.add_argument("--estimate", action="store_true", help="print resource estimate and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = EvalConfig()
    if args.batch_size:
        cfg.batch_size = int(args.batch_size)

    if args.smoke:
        return run_smoke(args, cfg)
    if args.estimate:
        cmd_estimate(args, cfg)
        return 0
    if args.stage is None:
        build_parser().print_help()
        return 2

    args.out = Path(args.out)
    if args.sites is None:
        args.sites = detect_sites(Path(args.data_root))
        if not args.sites:
            # Allow ablate/report over caches even when the dataset is gone.
            args.sites = [p.name for p in sorted(Path(args.out).glob("[0-9][0-9]")) if p.is_dir()]
    if not args.sites:
        print("No sites found (dataset missing and no cached outputs). See --estimate / README.")
        return 2
    print(f"sites: {', '.join(args.sites)}")

    if args.stage in ("build", "all"):
        stage_build(args, cfg)
    if args.stage in ("query", "all"):
        if args.stage == "all":
            for variant_args in (
                {"variant": "main"},
                {"variant": "no_heading"},
                {"variant": "heading_noise", "heading_noise": 5.0},
                {"variant": "heading_noise", "heading_noise": 10.0},
                {"variant": "heading_noise", "heading_noise": 15.0},
                {"variant": "gsd_error", "gsd_error": 0.1},
                {"variant": "gsd_error", "gsd_error": -0.1},
                {"variant": "gsd_error", "gsd_error": 0.2},
                {"variant": "gsd_error", "gsd_error": -0.2},
            ):
                for key, val in variant_args.items():
                    setattr(args, key.replace("-", "_"), val)
                stage_query(args, cfg)
        else:
            stage_query(args, cfg)
    if args.stage in ("ablate", "all"):
        stage_ablate(args, cfg)
    if args.stage in ("report", "all"):
        stage_report(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
