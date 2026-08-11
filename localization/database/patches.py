"""Step 5a - split a large satellite image into overlapping patches.

Paper §4.5: 500 px patch size, 100 px stride.

Each patch yields its top-left (x, y) in the parent image's pixel coordinates
so that descriptor centroids can be mapped back to the full satellite frame.
"""

from __future__ import annotations

import os
from typing import Iterator, Tuple
import itertools
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
    max_patches: int = None
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
    
    # Tính toán danh sách tọa độ x, y
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if not xs or xs[-1] != max(w - patch_size, 0):
        xs.append(max(w - patch_size, 0))
        
    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    if not ys or ys[-1] != max(h - patch_size, 0):
        ys.append(max(h - patch_size, 0))

    # Kết hợp các cặp tọa độ (y, x) đi kèm index của chúng
    # Kỹ thuật này giúp chuyển 2 vòng lặp for lồng nhau thành 1 vòng duy nhất
    grid = itertools.product(enumerate(ys), enumerate(xs))

    for count, ((row_idx, y), (col_idx, x)) in enumerate(grid):
        # Kiểm tra điều kiện dừng ngay lập tức nếu vượt quá max_patches
        if max_patches is not None and count >= max_patches:
            break
            
        patch = image.crop((x, y, x + patch_size, y + patch_size))
        patch_id = f"{tif_stem}_{row_idx:04d}_{col_idx:04d}"
        yield patch, patch_id, (x, y)