"""Step 5a - build per-patch satellite descriptors.

Both the **satellite** side (many sliding patches) and the **UAV** side (one
preprocessed image) go through the SAME ``segment_batch`` function at 500x500
to keep segmentation quality fair. The only difference is the batch size:
satellite uses ``batch_size > 1`` to amortize forward-pass overhead across
~25k patches, while UAV passes a single image.

The active descriptor is CFBVM-PF: one 24-D building shape vector per
Douglas-Peucker contour vertex.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from ..geometry.descriptor import (
    CFBVM_DESCRIPTOR_DIM,
    CFBVM_FEATURE_COLUMNS,
    CFBVM_REFERENCE_RADII_M,
    DEFAULT_METERS_PER_PIXEL,
    building_shape_vectors_from_polygons,
)
from ..segmentation.inference import segment_batch
from .patches import iter_patches, patch_count


def _polygons_to_descriptors(
    polygons: List[List[List[float]]],
    offset_xy: Tuple[float, float] = (0.0, 0.0),
    max_depth: int = 4,
    reference_radii_m: Tuple[float, float, float] = CFBVM_REFERENCE_RADII_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 24-D CFBVM-PF descriptors + global anchor coordinates."""
    _ = max_depth  # Ekeland-only legacy option; kept for API compatibility.
    off_x, off_y = float(offset_xy[0]), float(offset_xy[1])
    desc, anchors = building_shape_vectors_from_polygons(
        polygons,
        reference_radii_m=reference_radii_m,
        meters_per_pixel=meters_per_pixel,
    )
    if desc.shape[0] == 0:
        return np.zeros((0, CFBVM_DESCRIPTOR_DIM), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    anchors = anchors.copy()
    anchors[:, 0] += off_x
    anchors[:, 1] += off_y
    return desc, anchors


def extract_patch_descriptors(
    patch_image: Image.Image,
    model,
    device,
    score_threshold: float = 0.5,
    min_polygon_area: float = 50.0,
    tolerance_px: float = 2.0,
    max_depth: int = 4,
    reference_radii_m: Tuple[float, float, float] = CFBVM_REFERENCE_RADII_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single image through Steps 2-4 and return ``(descriptors, centroids)``.

    Centroids are in input-image-local pixel coordinates; the caller adds the
    patch's top-left offset when stitching into a multi-patch DB. For CFBVM-PF
    these are descriptor anchor vertices rather than triangle centroids. Used
    by the UAV query path (Notebook 2) and as a fallback in tests.
    ``tolerance_px`` is the Douglas-Peucker tolerance (paper §4.2, default
    2 px). ``max_depth`` is ignored and kept for old Ekeland-based callers.
    """
    from ..segmentation.inference import segment_image  # local import to keep torch lazy

    _binary_mask, polygons = segment_image(
        image=patch_image,
        model=model,
        device=device,
        score_threshold=score_threshold,
        min_area=min_polygon_area,
        tolerance_px=tolerance_px,
    )
    if not polygons:
        return np.zeros((0, CFBVM_DESCRIPTOR_DIM), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return _polygons_to_descriptors(
        polygons,
        max_depth=max_depth,
        reference_radii_m=reference_radii_m,
        meters_per_pixel=meters_per_pixel,
    )


def build_satellite_descriptors(
    tif_path: str,
    model,
    device,
    patch_size: int = 500,
    stride: int = 100,
    score_threshold: float = 0.5,
    min_polygon_area: float = 50.0,
    tolerance_px: float = 2.0,
    max_depth: int = 4,
    reference_radii_m: Tuple[float, float, float] = CFBVM_REFERENCE_RADII_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
    batch_size: int = 4,
    output_csv: Optional[str] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Iterate patches of a satellite GeoTIFF and collect 24-D CFBVM descriptors.

    Mask R-CNN inference is batched (``batch_size`` patches per forward pass)
    to amortize the cost of running the model. All patches go through the
    same ``segment_batch`` function used for the UAV image, so segmentation
    quality is identical on both branches.

    Returns a DataFrame with columns:
        ``patch_id, top_left_x, top_left_y, centroid_x, centroid_y,
         s11, ..., s38, parent_tif``

    ``centroid_x`` and ``centroid_y`` are the CFBVM vertex anchor coordinates in
    the parent satellite-image pixel coordinate system. ``max_depth`` is ignored
    and kept for old Ekeland-based callers.
    """
    Image.MAX_IMAGE_PIXELS = None
    parent_tif = os.path.basename(tif_path)
    image = Image.open(tif_path).convert("RGB")
    total_patches = patch_count(image.size, patch_size=patch_size, stride=stride)

    # Stream patches in batches without materializing the whole list.
    records: List[dict] = []
    iterator = iter_patches(image, patch_size=patch_size, stride=stride)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=total_patches, desc=f"Patches of {parent_tif}")
        except ImportError:
            pass

    pending_patches: List[Image.Image] = []
    pending_meta: List[Tuple[str, int, int]] = []  # (patch_id, tl_x, tl_y)

    def _flush(patches, meta):
        if not patches:
            return
        results = segment_batch(
            images=patches,
            model=model,
            device=device,
            score_threshold=score_threshold,
            min_area=min_polygon_area,
            tolerance_px=tolerance_px,
            batch_size=len(patches),  # already a chunk
        )
        for (_binary, polygons), (pid, tx, ty) in zip(results, meta):
            if not polygons:
                continue
            desc, cent = _polygons_to_descriptors(
                polygons,
                offset_xy=(tx, ty),
                max_depth=max_depth,
                reference_radii_m=reference_radii_m,
                meters_per_pixel=meters_per_pixel,
            )
            for d, c in zip(desc, cent):
                record = {
                    "patch_id": pid,
                    "parent_tif": parent_tif,
                    "top_left_x": int(tx),
                    "top_left_y": int(ty),
                    "centroid_x": float(c[0]),
                    "centroid_y": float(c[1]),
                }
                record.update({name: float(value) for name, value in zip(CFBVM_FEATURE_COLUMNS, d)})
                records.append(record)

    batch_size = max(1, int(batch_size))
    for patch_pil, patch_id, (tl_x, tl_y) in iterator:
        pending_patches.append(patch_pil)
        pending_meta.append((patch_id, tl_x, tl_y))
        if len(pending_patches) >= batch_size:
            _flush(pending_patches, pending_meta)
            pending_patches = []
            pending_meta = []
    _flush(pending_patches, pending_meta)

    df = pd.DataFrame.from_records(
        records,
        columns=[
            "patch_id",
            "parent_tif",
            "top_left_x",
            "top_left_y",
            "centroid_x",
            "centroid_y",
            *CFBVM_FEATURE_COLUMNS,
        ],
    )

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df
