"""Step 5a - build per-patch satellite descriptors (paper §4.5).

Both the **satellite** side (many sliding patches) and the **UAV** side (one
preprocessed image) go through the SAME ``segment_batch`` function at 500x500
to keep segmentation quality fair. The only difference is the batch size:
satellite uses ``batch_size > 1`` to amortize forward-pass overhead across
~25k patches, while UAV passes a single image.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from ..geometry.descriptor import triangle_descriptors_from_polygon
from ..segmentation.inference import segment_batch
from .patches import iter_patches, patch_count


def _polygons_to_descriptors(
    polygons: List[List[List[float]]],
    offset_xy: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate each polygon, compute 5-D descriptors + global centroids."""
    all_desc: List[np.ndarray] = []
    all_cent: List[np.ndarray] = []
    off_x, off_y = float(offset_xy[0]), float(offset_xy[1])
    for polygon in polygons:
        desc, cent = triangle_descriptors_from_polygon(polygon)
        if desc.shape[0] == 0:
            continue
        all_desc.append(desc)
        cent = cent.copy()
        cent[:, 0] += off_x
        cent[:, 1] += off_y
        all_cent.append(cent)
    if not all_desc:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.vstack(all_desc), np.vstack(all_cent)


def extract_patch_descriptors(
    patch_image: Image.Image,
    model,
    device,
    score_threshold: float = 0.5,
    min_polygon_area: float = 50.0,
    epsilon_factor: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single image through Steps 2-4 and return ``(descriptors, centroids)``.

    Centroids are in input-image-local pixel coordinates; the caller adds the
    patch's top-left offset when stitching into a multi-patch DB. Used by the
    UAV query path (Notebook 2) and as a fallback in tests.
    """
    from ..segmentation.inference import segment_image  # local import to keep torch lazy

    _binary_mask, polygons = segment_image(
        image=patch_image,
        model=model,
        device=device,
        score_threshold=score_threshold,
        min_area=min_polygon_area,
        tolerance_px=epsilon_factor,
    )
    if not polygons:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return _polygons_to_descriptors(polygons)


def build_satellite_descriptors(
    tif_path: str,
    model,
    device,
    patch_size: int = 500,
    stride: int = 100,
    score_threshold: float = 0.5,
    min_polygon_area: float = 50.0,
    epsilon_factor: float = 0.02,
    batch_size: int = 4,
    output_csv: Optional[str] = None,
    progress: bool = True,
    chunk_size_batches: int = 5000,
    resume_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Iterate patches of a satellite GeoTIFF and collect 5-D descriptors.

    Mask R-CNN inference is batched (``batch_size`` patches per forward pass)
    to amortize the cost of running the model. All patches go through the
    same ``segment_batch`` function used for the UAV image, so segmentation
    quality is identical on both branches.

    When ``resume_dir`` is set, the function writes chunk CSV files and a JSON
    metadata file so interrupted runs can resume from the last completed chunk.

    Returns a DataFrame with columns:
        ``patch_id, top_left_x, top_left_y, centroid_x, centroid_y,
         alpha1, alpha2, e1, e2, e3, parent_tif``

    ``centroid_x`` and ``centroid_y`` are in the parent satellite-image pixel
    coordinate system.
    """
    Image.MAX_IMAGE_PIXELS = None
    parent_tif = os.path.basename(tif_path)
    image = Image.open(tif_path).convert("RGB")
    total_patches = patch_count(image.size, patch_size=patch_size, stride=stride)

    chunk_size_batches = max(1, int(chunk_size_batches))
    batch_size = max(1, int(batch_size))
    resume_path = Path(resume_dir) if resume_dir else None
    metadata_path = resume_path / "build_satellite_descriptors.json" if resume_path else None
    final_output_path = Path(output_csv) if output_csv else None
    if resume_path:
        resume_path.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    chunk_files: List[str] = []
    completed_batches = 0
    completed_patches = 0
    chunk_index = 0
    finished = False

    def _write_metadata():
        if not metadata_path:
            return
        metadata = {
            "tif_path": os.path.abspath(tif_path),
            "parent_tif": parent_tif,
            "patch_size": patch_size,
            "stride": stride,
            "score_threshold": score_threshold,
            "min_polygon_area": min_polygon_area,
            "epsilon_factor": epsilon_factor,
            "batch_size": batch_size,
            "chunk_size_batches": chunk_size_batches,
            "total_patches": total_patches,
            "completed_patches": completed_patches,
            "completed_batches": completed_batches,
            "chunk_index": chunk_index,
            "chunk_files": chunk_files,
            "finished": finished,
            "output_csv": os.path.abspath(output_csv) if output_csv else None,
            "final_output_csv": os.path.abspath(output_csv) if output_csv else None,
        }
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        chunk_files = list(metadata.get("chunk_files", []))
        completed_patches = int(metadata.get("completed_patches", 0))
        completed_batches = int(metadata.get("completed_batches", 0))
        chunk_index = int(metadata.get("chunk_index", len(chunk_files)))
        finished = bool(metadata.get("finished", False))
        if finished and final_output_path and final_output_path.exists():
            print(f"Already finished. Loading from: {final_output_path}")
            return pd.read_csv(final_output_path)
        if completed_patches > 0:
            print(f"Resuming from patch {completed_patches}/{total_patches} (batch {completed_batches}, chunk {chunk_index})")

    # Stream patches in batches without materializing the whole list.
    iterator = iter_patches(image, patch_size=patch_size, stride=stride)

    # Skip already completed patches
    patches_to_skip = completed_patches
    for _ in range(patches_to_skip):
        try:
            next(iterator)
        except StopIteration:
            break

    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=total_patches - completed_patches, desc=f"Patches of {parent_tif}")
        except ImportError:
            pass

    pending_patches: List[Image.Image] = []
    pending_meta: List[Tuple[str, int, int]] = []  # (patch_id, tl_x, tl_y)

    def _flush(patches, meta):
        nonlocal records, completed_batches, completed_patches, chunk_index
        if not patches:
            return
        results = segment_batch(
            images=patches,
            model=model,
            device=device,
            score_threshold=score_threshold,
            min_area=min_polygon_area,
            tolerance_px=epsilon_factor,
            batch_size=len(patches),
        )
        for (_binary, polygons), (pid, tx, ty) in zip(results, meta):
            if not polygons:
                continue
            desc, cent = _polygons_to_descriptors(polygons, offset_xy=(tx, ty))
            for d, c in zip(desc, cent):
                records.append(
                    {
                        "patch_id": pid,
                        "parent_tif": parent_tif,
                        "top_left_x": int(tx),
                        "top_left_y": int(ty),
                        "centroid_x": float(c[0]),
                        "centroid_y": float(c[1]),
                        "alpha1": float(d[0]),
                        "alpha2": float(d[1]),
                        "e1": float(d[2]),
                        "e2": float(d[3]),
                        "e3": float(d[4]),
                    }
                )
        completed_batches += 1
        completed_patches += len(patches)
        if completed_batches % chunk_size_batches == 0:
            chunk_df = pd.DataFrame.from_records(
                records,
                columns=[
                    "patch_id",
                    "parent_tif",
                    "top_left_x",
                    "top_left_y",
                    "centroid_x",
                    "centroid_y",
                    "alpha1",
                    "alpha2",
                    "e1",
                    "e2",
                    "e3",
                ],
            )
            chunk_path = None
            if resume_path:
                chunk_path = resume_path / f"{Path(parent_tif).stem}_chunk_{chunk_index:05d}.csv"
                chunk_df.to_csv(chunk_path, index=False)
                chunk_files.append(str(chunk_path))
                records = []
                chunk_index += 1
                _write_metadata()
            elif output_csv:
                chunk_files.append(str(final_output_path))

    for patch_pil, patch_id, (tl_x, tl_y) in iterator:
        pending_patches.append(patch_pil)
        pending_meta.append((patch_id, tl_x, tl_y))

        if len(pending_patches) >= batch_size:
            _flush(pending_patches, pending_meta)
            pending_patches = []
            pending_meta = []
    _flush(pending_patches, pending_meta)

    if resume_path and records:
        chunk_df = pd.DataFrame.from_records(
            records,
            columns=[
                "patch_id",
                "parent_tif",
                "top_left_x",
                "top_left_y",
                "centroid_x",
                "centroid_y",
                "alpha1",
                "alpha2",
                "e1",
                "e2",
                "e3",
            ],
        )
        chunk_path = resume_path / f"{Path(parent_tif).stem}_chunk_{chunk_index:05d}.csv"
        chunk_df.to_csv(chunk_path, index=False)
        chunk_files.append(str(chunk_path))
        records = []
        chunk_index += 1
        _write_metadata()

    data_frames = []
    if resume_path:
        if records:
            chunk_df = pd.DataFrame.from_records(
                records,
                columns=[
                    "patch_id",
                    "parent_tif",
                    "top_left_x",
                    "top_left_y",
                    "centroid_x",
                    "centroid_y",
                    "alpha1",
                    "alpha2",
                    "e1",
                    "e2",
                    "e3",
                ],
            )
            chunk_path = resume_path / f"{Path(parent_tif).stem}_chunk_{chunk_index:05d}.csv"
            chunk_df.to_csv(chunk_path, index=False)
            chunk_files.append(str(chunk_path))
            records = []
            chunk_index += 1
            _write_metadata()

        for chunk_file in chunk_files:
            data_frames.append(pd.read_csv(chunk_file))
        if final_output_path:
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame(columns=[
            "patch_id",
            "parent_tif",
            "top_left_x",
            "top_left_y",
            "centroid_x",
            "centroid_y",
            "alpha1",
            "alpha2",
            "e1",
            "e2",
            "e3",
        ])
        finished = True
        _write_metadata()
        if final_output_path:
            df.to_csv(final_output_path, index=False)
        return df

    df = pd.DataFrame.from_records(
        records,
        columns=[
            "patch_id",
            "parent_tif",
            "top_left_x",
            "top_left_y",
            "centroid_x",
            "centroid_y",
            "alpha1",
            "alpha2",
            "e1",
            "e2",
            "e3",
        ],
    )

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df
