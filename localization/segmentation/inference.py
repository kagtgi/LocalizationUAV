"""Mask R-CNN inference + polygon extraction (paper §4.2).

The model is trained at 500x500 resolution. To keep segmentation quality
consistent for both UAV (one preprocessed image) and satellite (many sliding
patches), both paths call the SAME function (``segment_batch``) at the SAME
input size. The only difference is how many images are passed per call.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .contours import (
    contour_to_polygon,
    extract_contours_from_mask,
    filter_large_polygons_dynamic,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TO_TENSOR = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def _to_tensor(image: Image.Image, device) -> torch.Tensor:
    return _TO_TENSOR(image).to(device)


def _postprocess_one(
    pred: dict,
    image_size: Tuple[int, int],
    score_threshold: float,
    min_area: float,
    epsilon_factor: float,
    max_size_quantile: float,
    contour_method: str,
) -> Tuple[np.ndarray, List[List[List[float]]]]:
    """Convert one model prediction dict into ``(binary_mask, polygons)``."""
    w, h = image_size
    scores = pred["scores"].detach().cpu().numpy()
    keep = scores >= float(score_threshold)
    if not keep.any():
        return np.zeros((h, w), dtype=np.uint8), []

    masks = pred["masks"][keep].detach().cpu().numpy()
    combined = np.zeros((h, w), dtype=np.float32)
    for m in masks:
        mm = m[0] if m.ndim == 3 else m
        combined = np.maximum(combined, mm.astype(np.float32))
    binary_mask = (combined > 0.5).astype(np.uint8)

    contours = extract_contours_from_mask(binary_mask, min_area=min_area, method=contour_method)
    polygons = [contour_to_polygon(c, epsilon_factor=epsilon_factor) for c in contours]
    contours, polygons = filter_large_polygons_dynamic(
        contours, polygons, image_shape=(h, w), quantile=max_size_quantile
    )
    polygons = [p for p in polygons if len(p) >= 3]
    return binary_mask, polygons


def segment_batch(
    images: Sequence[Image.Image],
    model,
    device,
    score_threshold: float = 0.5,
    min_area: float = 50.0,
    epsilon_factor: float = 0.02,
    max_size_quantile: float = 0.995,
    contour_method: str = "marching_squares",
    batch_size: int = 4,
) -> List[Tuple[np.ndarray, List[List[List[float]]]]]:
    """Run Mask R-CNN on a sequence of images using mini-batches.

    Same per-image post-processing for every image (UAV or satellite patch),
    so segmentation quality is identical between the two branches. Returns
    one ``(binary_mask, polygons)`` per input image, in order.
    """
    if not images:
        return []

    batch_size = max(1, int(batch_size))
    out: List[Tuple[np.ndarray, List[List[List[float]]]]] = []

    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors = [_to_tensor(img, device) for img in chunk]
        with torch.no_grad():
            predictions = model(tensors)
        for image, pred in zip(chunk, predictions):
            out.append(
                _postprocess_one(
                    pred=pred,
                    image_size=image.size,
                    score_threshold=score_threshold,
                    min_area=min_area,
                    epsilon_factor=epsilon_factor,
                    max_size_quantile=max_size_quantile,
                    contour_method=contour_method,
                )
            )
    return out


def segment_image(
    image: Image.Image,
    model,
    device,
    score_threshold: float = 0.5,
    min_area: float = 50.0,
    epsilon_factor: float = 0.02,
    max_size_quantile: float = 0.995,
    contour_method: str = "marching_squares",
) -> Tuple[np.ndarray, List[List[List[float]]]]:
    """Single-image wrapper around :func:`segment_batch` (batch_size=1)."""
    results = segment_batch(
        images=[image],
        model=model,
        device=device,
        score_threshold=score_threshold,
        min_area=min_area,
        epsilon_factor=epsilon_factor,
        max_size_quantile=max_size_quantile,
        contour_method=contour_method,
        batch_size=1,
    )
    return results[0] if results else (np.zeros((image.size[1], image.size[0]), dtype=np.uint8), [])


# -------- sliding-window full-image inference (kept for whole-mask use cases) --------


def _pyramid_blend_weight(patch_size: int) -> torch.Tensor:
    """Pyramid (hat-shaped) blending weight for smooth patch stitching."""
    x = torch.linspace(0, 1, patch_size)
    y = torch.linspace(0, 1, patch_size)
    weight_x = 1 - torch.abs(2 * x - 1)
    weight_y = 1 - torch.abs(2 * y - 1)
    weight = weight_x.view(1, -1) * weight_y.view(-1, 1)
    return torch.clamp(weight, min=0.001)


def segment_image_patchwise(
    image: Image.Image,
    model,
    device,
    patch_size: int = 500,
    overlap: int = 100,
    score_threshold: float = 0.5,
    batch_size: int = 4,
    progress: bool = True,
) -> np.ndarray:
    """Sliding-window Mask R-CNN over a large image, returning a soft mask.

    Used only when a *full-size satellite mask* is required (e.g. for the
    legacy whole-image-vs-UAV-mask comparison). The standard Step 5a pipeline
    does NOT call this: it uses :func:`segment_batch` per patch and keeps
    per-patch polygons + descriptors.
    """
    orig_w, orig_h = image.size
    pad_w = (patch_size - orig_w % patch_size) % patch_size
    pad_h = (patch_size - orig_h % patch_size) % patch_size
    padded_w, padded_h = orig_w + pad_w, orig_h + pad_h

    if pad_w or pad_h:
        padded = Image.new("RGB", (padded_w, padded_h))
        padded.paste(image, (0, 0))
    else:
        padded = image

    full_mask_sum = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    full_weight_sum = torch.zeros((padded_h, padded_w), dtype=torch.float32)

    stride = patch_size - overlap
    coords = [(t, l) for t in range(0, padded_h, stride) for l in range(0, padded_w, stride)]
    weight = _pyramid_blend_weight(patch_size)

    iterator: Iterable = range(0, len(coords), max(1, int(batch_size)))
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(list(iterator), desc=f"Soft mask {orig_w}x{orig_h}", leave=False)
        except ImportError:
            pass

    for batch_start in iterator:
        chunk_coords = coords[batch_start : batch_start + batch_size]
        chunk_coords = [
            (min(t, padded_h - patch_size), min(l, padded_w - patch_size))
            for t, l in chunk_coords
        ]
        chunk_patches = [
            padded.crop((l, t, l + patch_size, t + patch_size))
            for t, l in chunk_coords
        ]
        tensors = [_to_tensor(p, device) for p in chunk_patches]
        with torch.no_grad():
            predictions = model(tensors)

        for (top, left), pred in zip(chunk_coords, predictions):
            combined = torch.zeros((patch_size, patch_size), dtype=torch.float32)
            if pred["masks"].numel() > 0:
                scores = pred["scores"].detach().cpu()
                masks = pred["masks"].detach().cpu()
                for i in range(masks.shape[0]):
                    if scores[i] >= score_threshold:
                        combined = torch.maximum(combined, masks[i, 0])
            full_mask_sum[top : top + patch_size, left : left + patch_size] += combined * weight
            full_weight_sum[top : top + patch_size, left : left + patch_size] += weight

    soft_mask = (full_mask_sum / (full_weight_sum + 1e-6)).clamp(0, 1).numpy()
    return soft_mask[:orig_h, :orig_w]


def polygons_from_soft_mask(
    soft_mask: np.ndarray,
    score_threshold: float = 0.5,
    min_area: float = 50.0,
    epsilon_factor: float = 0.02,
    max_size_quantile: float = 0.995,
    contour_method: str = "marching_squares",
) -> Tuple[np.ndarray, List[List[List[float]]]]:
    """Threshold a soft mask and extract Douglas-Peucker polygons."""
    binary_mask = (soft_mask > float(score_threshold)).astype(np.uint8)
    h, w = binary_mask.shape[:2]
    contours = extract_contours_from_mask(binary_mask, min_area=min_area, method=contour_method)
    polygons = [contour_to_polygon(c, epsilon_factor=epsilon_factor) for c in contours]
    contours, polygons = filter_large_polygons_dynamic(
        contours, polygons, image_shape=(h, w), quantile=max_size_quantile
    )
    polygons = [p for p in polygons if len(p) >= 3]
    return binary_mask, polygons
