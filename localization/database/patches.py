"""Step 5a - split a large satellite image into overlapping patches.

Paper §4.5: 500 px patch size, 100 px stride.

Each patch yields its top-left (x, y) in the parent image's pixel coordinates
so that descriptor centroids can be mapped back to the full satellite frame.
"""

from __future__ import annotations

import math
import os
from typing import Iterator, Optional, Tuple

from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def patch_count(image_size: Tuple[int, int], patch_size: int = 500, stride: int = 100) -> int:
    """Return the number of patches a full sliding-window pass would produce."""
    w, h = image_size
    cols = max(1, (max(w - patch_size, 0) + stride - 1) // stride + 1)
    rows = max(1, (max(h - patch_size, 0) + stride - 1) // stride + 1)
    return cols * rows


def iter_patches(
    image_or_path,
    patch_size: int = 500,
    stride: int = 100,
) -> Iterator[Tuple[Image.Image, str, Tuple[int, int]]]:
    """Yield ``(patch_pil, patch_id, (top_left_x, top_left_y))`` over a large image.

    The image is loaded once and patches are cropped lazily. The last row/col
    are anchored to ``image_size - patch_size`` so they're always full-size.
    """
    if isinstance(image_or_path, (str, bytes, os.PathLike)):
        tif_stem = os.path.splitext(os.path.basename(str(image_or_path)))[0]
        image = Image.open(image_or_path).convert("RGB")
    else:
        tif_stem = "image"
        image = image_or_path

    w, h = image.size
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if not xs or xs[-1] != max(w - patch_size, 0):
        xs.append(max(w - patch_size, 0))
    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    if not ys or ys[-1] != max(h - patch_size, 0):
        ys.append(max(h - patch_size, 0))

    for row_idx, y in enumerate(ys):
        for col_idx, x in enumerate(xs):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patch_id = f"{tif_stem}_{row_idx:04d}_{col_idx:04d}"
            yield patch, patch_id, (x, y)


def patch_owns_centroid(
    centroid_xy: Tuple[float, float],
    patch_tl_xy: Tuple[float, float],
    patch_size: int,
    stride: int,
    cell_size: Optional[float] = None,
) -> bool:
    """True if ``centroid_xy`` falls in the ``cell_size`` x ``cell_size`` box
    centered on this patch.

    At patch_size=500 / stride=100, adjacent patches overlap 80%, so without
    any ownership check the same physical building is independently
    re-segmented and recorded once per overlapping patch it falls inside
    (empirically ~15-25x redundancy) - enough that coincidentally
    near-identical descriptors from unrelated buildings elsewhere in a
    multi-million-triangle database out-compete the true match.

    ``cell_size`` trades off redundancy against triangles-per-patch: at
    ``cell_size=stride`` (the default), cells exactly tile the image with no
    overlap and no gaps (a triangle has exactly one owner - the patch whose
    center it is nearest to - since patches sit on a regular stride-spaced
    grid), which eliminates duplication entirely but can leave very few
    triangles per patch, weakening plurality-vote concentration. A wider
    ``cell_size`` (still less than ``patch_size``) re-admits bounded overlap
    among immediately-neighboring patches in exchange for more triangles per
    voting bucket.
    """
    if cell_size is None:
        cell_size = float(stride)
    cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    px = float(patch_tl_xy[0]) + patch_size / 2.0
    py = float(patch_tl_xy[1]) + patch_size / 2.0
    half = cell_size / 2.0
    # Half-open on the upper bound so that at cell_size=stride exactly (the
    # zero-overlap case), a point exactly on a cell boundary is owned by
    # exactly one neighbor, never both or neither.
    return (-half <= cx - px < half) and (-half <= cy - py < half)


def vote_bucket_id(centroid_xy: Tuple[float, float], bucket_size: float, prefix: str = "") -> str:
    """Coarse voting-bucket identifier for a (deduplicated) triangle centroid.

    Deduplication (patch_owns_centroid) removes redundant copies of the same
    physical triangle, but its ownership cell must stay narrow to actually
    eliminate overlap, which leaves few triangles per original sliding-window
    patch - too few for the plurality vote to concentrate reliably, since a
    single UAV query's field of view spans several such cells. The voting
    *bucket* is a separate, independent grid (typically bucket_size =
    patch_size, matching one UAV query's approximate ground footprint): every
    deduplicated triangle whose centroid falls in the same bucket_size x
    bucket_size cell shares one identity for plurality-vote and position-
    output purposes, regardless of which original patch it was segmented in.
    """
    bx = int(math.floor(float(centroid_xy[0]) / bucket_size))
    by = int(math.floor(float(centroid_xy[1]) / bucket_size))
    return f"{prefix}vb_{bx:05d}_{by:05d}"
