"""Polygon extraction utilities (paper §4.2 - Douglas-Peucker simplification)."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from skimage import measure


def extract_contours_from_mask(mask: np.ndarray, min_area: float = 100.0, method: str = "marching_squares") -> List[np.ndarray]:
    """Extract building contours from a binary mask via OpenCV or marching squares."""
    valid_contours: List[np.ndarray] = []

    if method == "opencv":
        if mask.max() <= 1:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                valid_contours.append(cnt)
        return valid_contours

    if method == "marching_squares":
        if mask.dtype == np.uint8:
            if mask.max() <= 1:
                mask_float = mask.astype(np.float32)
            else:
                mask_float = mask.astype(np.float32) / 255.0
        else:
            mask_float = mask
        level = 0.5
        raw_contours = measure.find_contours(mask_float, level=level)
        for cnt in raw_contours:
            cnt_xy = np.fliplr(cnt)
            cnt_formatted = cnt_xy.reshape(-1, 1, 2).astype(np.float32)
            if cv2.contourArea(cnt_formatted) >= min_area:
                valid_contours.append(cnt_formatted)
        return valid_contours

    raise ValueError(f"Unknown contour method: {method!r}")


def contour_to_polygon(contour: np.ndarray, epsilon_factor: float = 0.02) -> List[List[float]]:
    """Douglas-Peucker simplification of a contour (paper §4.2, tau=2px ~ 0.02 of perimeter)."""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon.reshape(-1, 2).tolist()


def contour_to_polygon_dynamic(
    contour: np.ndarray,
    image_shape: Tuple[int, int],
    min_epsilon_factor: float = 0.002,
    max_epsilon_factor: float = 0.02,
) -> List[List[float]]:
    """Adaptive Douglas-Peucker: small contours keep more detail."""
    img_h, img_w = image_shape[:2] if len(image_shape) == 3 else image_shape
    image_area = max(float(img_h * img_w), 1.0)
    area = max(float(cv2.contourArea(contour)), 1.0)
    area_ratio = float(np.clip(area / image_area, 0.0, 1.0))
    t = np.sqrt(area_ratio)
    epsilon_factor = min_epsilon_factor + (max_epsilon_factor - min_epsilon_factor) * t
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon.reshape(-1, 2).tolist()


def filter_large_polygons(
    contours: List[np.ndarray],
    polygons: List[List[List[float]]],
    image_shape: Tuple[int, int],
    max_size_ratio: float = 0.20,
):
    """Drop polygons exceeding ``max_size_ratio`` of the image area (edge artifacts)."""
    img_h, img_w = image_shape[:2] if len(image_shape) == 3 else image_shape
    image_area = img_h * img_w
    max_allowed_area = image_area * max_size_ratio

    filtered_contours: List[np.ndarray] = []
    filtered_polygons: List[List[List[float]]] = []
    for contour, polygon in zip(contours, polygons):
        if cv2.contourArea(contour) < max_allowed_area:
            filtered_contours.append(contour)
            filtered_polygons.append(polygon)
    return filtered_contours, filtered_polygons


def filter_large_polygons_dynamic(
    contours: List[np.ndarray],
    polygons: List[List[List[float]]],
    image_shape: Tuple[int, int],
    quantile: float = 0.995,
    fallback_max_size_ratio: float = 0.95,
):
    """Drop only extreme-outlier polygons by area-quantile cap."""
    if not contours:
        return contours, polygons

    img_h, img_w = image_shape[:2] if len(image_shape) == 3 else image_shape
    image_area = float(max(img_h * img_w, 1))
    hard_cap = image_area * fallback_max_size_ratio

    areas = np.array([max(float(cv2.contourArea(c)), 0.0) for c in contours], dtype=np.float64)
    if len(areas) >= 4:
        q_cap = float(np.quantile(areas, quantile))
        max_allowed_area = min(max(q_cap, 1.0), hard_cap)
    else:
        max_allowed_area = hard_cap

    filtered_contours: List[np.ndarray] = []
    filtered_polygons: List[List[List[float]]] = []
    for contour, polygon, area in zip(contours, polygons, areas):
        if area <= max_allowed_area:
            filtered_contours.append(contour)
            filtered_polygons.append(polygon)
    return filtered_contours, filtered_polygons
