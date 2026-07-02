"""Step 5a - build per-patch satellite descriptors (paper §4.5).

Both the **satellite** side (many sliding patches) and the **UAV** side (one
preprocessed image) go through the SAME ``segment_batch`` function at 500x500
to keep segmentation quality fair. The only difference is the batch size:
satellite uses ``batch_size > 1`` to amortize forward-pass overhead across
~25k patches, while UAV passes a single image.
"""

from __future__ import annotations

import math
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from ..geometry.descriptor import triangle_descriptors_from_polygon
from ..segmentation.inference import segment_batch
from .patches import iter_patches, patch_count, patch_owns_centroid, vote_bucket_id


def _polygons_to_descriptors(
    polygons: List[List[List[float]]],
    offset_xy: Tuple[float, float] = (0.0, 0.0),
    max_depth: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate each polygon, compute 5-D descriptors + global centroids."""
    all_desc: List[np.ndarray] = []
    all_cent: List[np.ndarray] = []
    off_x, off_y = float(offset_xy[0]), float(offset_xy[1])
    for polygon in polygons:
        desc, cent = triangle_descriptors_from_polygon(polygon, max_depth=max_depth)
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
    tolerance_px: float = 2.0,
    max_depth: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single image through Steps 2-4 and return ``(descriptors, centroids)``.

    Centroids are in input-image-local pixel coordinates; the caller adds the
    patch's top-left offset when stitching into a multi-patch DB. Used by the
    UAV query path (Notebook 2) and as a fallback in tests. ``tolerance_px`` is
    the Douglas-Peucker tolerance (paper §4.2, default 2 px) and ``max_depth``
    is the kernel-expansion cap ``D_max`` (paper Algorithm 1, default 4).
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
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return _polygons_to_descriptors(polygons, max_depth=max_depth)


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
    batch_size: int = 4,
    output_csv: Optional[str] = None,
    progress: bool = True,
    polygon_sink: Optional[Callable[[str, Tuple[int, int], List[List[List[float]]]], None]] = None,
    max_patches: Optional[int] = None,
    dedup_cell_size: Optional[float] = None,
    vote_bucket_size: Optional[float] = None,
) -> pd.DataFrame:
    """Iterate patches of a satellite GeoTIFF and collect 5-D descriptors.

    Mask R-CNN inference is batched (``batch_size`` patches per forward pass)
    to amortize the cost of running the model. All patches go through the
    same ``segment_batch`` function used for the UAV image, so segmentation
    quality is identical on both branches.

    ``polygon_sink(patch_id, (tl_x, tl_y), polygons)`` is called once per patch
    that yields at least one polygon; ``eval.py`` uses it to cache segmentation
    output so descriptor ablations (e.g. the ``D_max`` sweep) can recompute
    descriptors without re-running Mask R-CNN. ``max_patches`` caps the number
    of patches processed (smoke testing only - NOT for paper results).

    ``dedup_cell_size`` (default ``stride``) and ``vote_bucket_size`` (default
    ``patch_size``) are two independent grids: ``dedup_cell_size`` decides
    which single overlapping patch "owns" a triangle (eliminating the ~15-25x
    redundant copies otherwise produced by the 80%-overlapping sliding
    window), while ``vote_bucket_size`` is the coarser grid actually used for
    the ``patch_id`` column - i.e. the plurality-vote and position-output
    identity - since the ownership cell is too narrow on its own for votes to
    reliably concentrate (a single UAV query's field of view spans several
    ownership cells). See ``localization.database.patches.vote_bucket_id``.

    Returns a DataFrame with columns:
        ``patch_id, top_left_x, top_left_y, centroid_x, centroid_y,
         alpha1, alpha2, e1, e2, e3, parent_tif``

    ``centroid_x`` and ``centroid_y`` are in the parent satellite-image pixel
    coordinate system. ``patch_id`` is the vote bucket (see above); when
    ``vote_bucket_size`` is left at its default, ``top_left_x/y`` are the
    bucket's own top-left corner (``bucket_index * vote_bucket_size``), so
    every row sharing a ``patch_id`` also shares the same ``top_left_x/y``.
    """
    Image.MAX_IMAGE_PIXELS = None
    parent_tif = os.path.basename(tif_path)
    bucket_size = float(vote_bucket_size) if vote_bucket_size is not None else float(patch_size)
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
            if polygon_sink is not None:
                polygon_sink(pid, (tx, ty), polygons)
            desc, cent = _polygons_to_descriptors(polygons, offset_xy=(tx, ty), max_depth=max_depth)
            for d, c in zip(desc, cent):
                # Deduplicate across the 80%-overlapping patch grid: only keep
                # a triangle in patches within dedup_cell_size of its centroid.
                if not patch_owns_centroid((c[0], c[1]), (tx, ty), patch_size, stride, cell_size=dedup_cell_size):
                    continue
                # Vote bucket (paper's actual voting/position-output identity)
                # is a separate, coarser grid than the dedup-ownership cell -
                # see the module-level docstring above.
                bucket_id = vote_bucket_id((c[0], c[1]), bucket_size, prefix=f"{parent_tif}_")
                bucket_tl_x = int(math.floor(c[0] / bucket_size) * bucket_size)
                bucket_tl_y = int(math.floor(c[1] / bucket_size) * bucket_size)
                records.append(
                    {
                        "patch_id": bucket_id,
                        "parent_tif": parent_tif,
                        "top_left_x": bucket_tl_x,
                        "top_left_y": bucket_tl_y,
                        "centroid_x": float(c[0]),
                        "centroid_y": float(c[1]),
                        "alpha1": float(d[0]),
                        "alpha2": float(d[1]),
                        "e1": float(d[2]),
                        "e2": float(d[3]),
                        "e3": float(d[4]),
                    }
                )

    batch_size = max(1, int(batch_size))
    n_seen = 0
    for patch_pil, patch_id, (tl_x, tl_y) in iterator:
        pending_patches.append(patch_pil)
        pending_meta.append((patch_id, tl_x, tl_y))
        n_seen += 1
        if len(pending_patches) >= batch_size:
            _flush(pending_patches, pending_meta)
            pending_patches = []
            pending_meta = []
        if max_patches is not None and n_seen >= int(max_patches):
            break
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
