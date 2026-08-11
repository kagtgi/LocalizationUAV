#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

MANIFEST_FIELDS = [
    "source_image",
    "patch_id",
    "row_idx",
    "col_idx",
    "x",
    "y",
    "w",
    "h",
    "center_x",
    "center_y",
    "subset",
    "file",
    "status",
    "assignee",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Cut a large satellite TIFF into patches and write manifest.csv."
    )
    p.add_argument("--satellite-tif", type=Path, required=True,
                   help="Path to the large satellite .tif")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output folder, e.g. data/cvat/sat01")
    p.add_argument("--patch-size", type=int, default=1024,
                   help="Patch size in pixels")
    p.add_argument("--stride", type=int, default=1024,
                   help="Stride in pixels. For annotation, stride=patch_size is recommended.")
    p.add_argument("--split-json", type=Path, default=None,
                   help=(
                       "Optional JSON to assign subset by patch center. "
                       "Format: {\"train\": [[x0,y0,x1,y1]], \"val\": [...], \"test\": [...]} "
                       "Coordinates are normalized in [0,1]."
                   ))
    p.add_argument("--manifest-name", type=str, default="manifest.csv")
    p.add_argument("--patch-dir-name", type=str, default="patches")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_split_rules(split_json: Path | None) -> Dict[str, List[Tuple[float, float, float, float]]]:
    if split_json is None:
        return {}
    data = json.loads(split_json.read_text(encoding="utf-8"))
    rules: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for subset, rects in data.items():
        rules[subset] = [tuple(map(float, r)) for r in rects]
    return rules


def assign_subset(cx: float, cy: float, w: int, h: int,
                  rules: Dict[str, List[Tuple[float, float, float, float]]]) -> str:
    if not rules:
        return ""
    rx = cx / max(w, 1)
    ry = cy / max(h, 1)
    for subset, rects in rules.items():
        for x0, y0, x1, y1 in rects:
            if x0 <= rx < x1 and y0 <= ry < y1:
                return subset
    return "unassigned"


def grid_positions(length: int, patch_size: int, stride: int) -> List[int]:
    limit = max(length - patch_size, 0)
    positions = list(range(0, limit + 1, stride))
    if not positions or positions[-1] != limit:
        positions.append(limit)
    return positions


def main():
    args = parse_args()

    sat_path = args.satellite_tif.resolve()
    out_dir = args.out_dir.resolve()
    patch_dir = out_dir / args.patch_dir_name
    manifest_path = out_dir / args.manifest_name

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists. Use --overwrite to replace it.")

    split_rules = load_split_rules(args.split_json)

    with Image.open(sat_path) as img, manifest_path.open("w", newline="", encoding="utf-8") as f:
        w, h = img.size
        xs = grid_positions(w, args.patch_size, args.stride)
        ys = grid_positions(h, args.patch_size, args.stride)

        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for row_idx, y in enumerate(ys):
            for col_idx, x in enumerate(xs):
                patch = img.crop((x, y, x + args.patch_size, y + args.patch_size))
                patch_id = f"{sat_path.stem}_x{x:06d}_y{y:06d}"
                patch_file = patch_dir / f"{patch_id}.tif"

                patch.save(str(patch_file), format="TIFF")

                center_x = x + args.patch_size / 2.0
                center_y = y + args.patch_size / 2.0
                subset = assign_subset(center_x, center_y, w, h, split_rules)

                writer.writerow({
                    "source_image": sat_path.name,
                    "patch_id": patch_id,
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "x": x,
                    "y": y,
                    "w": args.patch_size,
                    "h": args.patch_size,
                    "center_x": center_x,
                    "center_y": center_y,
                    "subset": subset,
                    "file": f"{patch_id}.tif",
                    "status": "unlabeled",
                    "assignee": "",
                })

    print(f"[OK] patches: {patch_dir}")
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
