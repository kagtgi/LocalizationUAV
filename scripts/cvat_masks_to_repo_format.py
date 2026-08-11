#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert CVAT-exported PNG masks to TIFF and organize repo folders."
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="manifest.csv created by prepare_satellite_patches.py")
    p.add_argument("--patch-dir", type=Path, required=True,
                   help="Folder containing patch .tif images")
    p.add_argument("--mask-dir", type=Path, required=True,
                   help="Folder containing CVAT-exported mask files")
    p.add_argument("--output-root", type=Path, required=True,
                   help='Output root, e.g. "GF-7 Building (3Bands)"')
    p.add_argument("--mask-ext", type=str, default=".png",
                   help="Mask extension exported by CVAT, e.g. .png")
    p.add_argument("--subset-col", type=str, default="subset")
    p.add_argument("--file-col", type=str, default="file")
    p.add_argument("--subsets", type=str, default="train,val,test")
    return p.parse_args()


def find_mask(mask_dir: Path, stem: str, mask_ext: str) -> Path | None:
    ext = mask_ext if mask_ext.startswith(".") else f".{mask_ext}"
    candidates = list(mask_dir.rglob(f"{stem}{ext}"))
    if candidates:
        return candidates[0]
    return None


def png_to_binary_tif(src_mask: Path, dst_mask: Path):
    arr = np.array(Image.open(src_mask).convert("L"), dtype=np.uint8)
    arr = np.where(arr > 0, 255, 0).astype(np.uint8)
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(dst_mask), arr)


def main():
    args = parse_args()

    df = pd.read_csv(args.manifest)
    allowed_subsets = {s.strip() for s in args.subsets.split(",") if s.strip()}

    for _, row in df.iterrows():
        subset = str(row.get(args.subset_col, "")).strip()
        if subset not in allowed_subsets:
            continue

        file_name = str(row[args.file_col]).strip()
        patch_path = args.patch_dir / file_name
        if not patch_path.exists():
            print(f"[WARN] missing patch: {patch_path}")
            continue

        stem = Path(file_name).stem
        mask_path = find_mask(args.mask_dir, stem, args.mask_ext)
        if mask_path is None:
            print(f"[WARN] missing mask for: {stem}")
            continue

        out_img_dir = args.output_root / subset / "image"
        out_lbl_dir = args.output_root / subset / "label"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(patch_path, out_img_dir / patch_path.name)
        png_to_binary_tif(mask_path, out_lbl_dir / f"{stem}.tif")

    print(f"[OK] output root: {args.output_root}")


if __name__ == "__main__":
    main()