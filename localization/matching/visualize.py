"""Visualization helpers for query results.

The headline figures are:

* :func:`draw_gt_and_prediction` - the full satellite image with a red X at
  the predicted position and a magenta star at the ground-truth position,
  joined by a yellow line annotated with the error distance in meters.
* :func:`render_localization_result` - a 3-panel layout combining (left) the
  full satellite map with both markers, (center) a zoom around the prediction
  showing the same markers, and (right) the UAV view.

``draw_predicted_position`` / ``render_match_figure`` are kept for
backward compatibility with the original two-panel layout.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..io.bounds import latlon_to_pixel


Image.MAX_IMAGE_PIXELS = None


# ===== colors (RGB tuples, matching the cv2/matplotlib RGB convention used here) =====
COLOR_PRED = (255, 0, 0)        # red X for predicted position
COLOR_GT = (255, 0, 255)        # magenta star for ground truth
COLOR_CONNECTOR = (255, 220, 0) # yellow line between GT and prediction
COLOR_LEGEND_BG = (255, 255, 255)
COLOR_LEGEND_TEXT = (32, 32, 32)


# ----------------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------------


def _put_labelled_marker(
    img_rgb: np.ndarray,
    xy: Tuple[float, float],
    marker_type: int,
    color: Tuple[int, int, int],
    label: str,
    marker_size: int,
    marker_thickness: int,
    font_scale: float,
    text_thickness: int,
) -> None:
    """Draw an OpenCV marker plus a labeled chip just above it (in-place)."""
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.drawMarker(
        img_rgb,
        (x, y),
        color=color,
        markerType=marker_type,
        markerSize=marker_size,
        thickness=marker_thickness,
    )
    # Label background chip
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    pad = max(4, int(round(font_scale * 6)))
    chip_w = tw + 2 * pad
    chip_h = th + baseline + 2 * pad
    chip_x = max(0, x - chip_w // 2)
    chip_y = max(0, y - marker_size // 2 - chip_h - pad)
    cv2.rectangle(
        img_rgb,
        (chip_x, chip_y),
        (chip_x + chip_w, chip_y + chip_h),
        COLOR_LEGEND_BG,
        thickness=-1,
    )
    cv2.rectangle(
        img_rgb,
        (chip_x, chip_y),
        (chip_x + chip_w, chip_y + chip_h),
        color,
        thickness=max(2, text_thickness),
    )
    cv2.putText(
        img_rgb,
        label,
        (chip_x + pad, chip_y + chip_h - pad - baseline // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        text_thickness,
        cv2.LINE_AA,
    )


def _scale_for_size(image_dim: int) -> Tuple[int, int, float, int]:
    """Pick marker_size, marker_thickness, font_scale, text_thickness for an image."""
    marker_size = max(120, min(400, image_dim // 60))
    marker_thickness = max(4, marker_size // 25)
    font_scale = max(1.0, marker_size / 200.0)
    text_thickness = max(2, int(round(font_scale * 2)))
    return marker_size, marker_thickness, font_scale, text_thickness


def draw_gt_and_prediction(
    satellite_image_path: str,
    predicted_pixel_xy: Tuple[float, float],
    gt_pixel_xy: Optional[Tuple[float, float]] = None,
    error_distance_m: Optional[float] = None,
    marker_size: Optional[int] = None,
    marker_thickness: Optional[int] = None,
) -> np.ndarray:
    """Render the full satellite image with prediction (red X) and optional GT
    (magenta star), connected by a yellow line annotated with the error.

    Returns the modified image as an RGB ``np.ndarray``.
    """
    img_bgr = cv2.imread(satellite_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read satellite image: {satellite_image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    auto_size, auto_thick, font_scale, text_thick = _scale_for_size(max(h, w))
    marker_size = int(marker_size or auto_size)
    marker_thickness = int(marker_thickness or auto_thick)

    pred_xy = (float(predicted_pixel_xy[0]), float(predicted_pixel_xy[1]))

    # Yellow connector line + midpoint label (drawn first so markers sit on top)
    if gt_pixel_xy is not None:
        gt_xy = (float(gt_pixel_xy[0]), float(gt_pixel_xy[1]))
        cv2.line(
            img_rgb,
            (int(round(gt_xy[0])), int(round(gt_xy[1]))),
            (int(round(pred_xy[0])), int(round(pred_xy[1]))),
            COLOR_CONNECTOR,
            thickness=max(2, marker_thickness // 2),
            lineType=cv2.LINE_AA,
        )
        if error_distance_m is not None:
            mid_x = int(round(0.5 * (gt_xy[0] + pred_xy[0])))
            mid_y = int(round(0.5 * (gt_xy[1] + pred_xy[1])))
            label = f"error = {error_distance_m:.1f} m"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
            pad = max(4, int(round(font_scale * 6)))
            cv2.rectangle(
                img_rgb,
                (mid_x - tw // 2 - pad, mid_y - th - baseline - pad),
                (mid_x + tw // 2 + pad, mid_y + pad),
                COLOR_LEGEND_BG,
                thickness=-1,
            )
            cv2.rectangle(
                img_rgb,
                (mid_x - tw // 2 - pad, mid_y - th - baseline - pad),
                (mid_x + tw // 2 + pad, mid_y + pad),
                COLOR_CONNECTOR,
                thickness=max(2, text_thick),
            )
            cv2.putText(
                img_rgb,
                label,
                (mid_x - tw // 2, mid_y - baseline // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                COLOR_LEGEND_TEXT,
                text_thick,
                cv2.LINE_AA,
            )

    # Markers
    _put_labelled_marker(
        img_rgb,
        pred_xy,
        marker_type=cv2.MARKER_TILTED_CROSS,
        color=COLOR_PRED,
        label="PRED",
        marker_size=marker_size,
        marker_thickness=marker_thickness,
        font_scale=font_scale,
        text_thickness=text_thick,
    )
    if gt_pixel_xy is not None:
        _put_labelled_marker(
            img_rgb,
            gt_pixel_xy,
            marker_type=cv2.MARKER_STAR,
            color=COLOR_GT,
            label="GT",
            marker_size=marker_size,
            marker_thickness=marker_thickness,
            font_scale=font_scale,
            text_thickness=text_thick,
        )

    return img_rgb


def crop_zoom_around(
    image_rgb: np.ndarray,
    predicted_pixel_xy: Tuple[float, float],
    gt_pixel_xy: Optional[Tuple[float, float]] = None,
    zoom_radius_px: int = 1500,
    min_radius_px: int = 600,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop a window around the prediction (and optionally containing the GT).

    Returns ``(crop_rgb, (offset_x, offset_y))`` where ``offset_*`` is the
    top-left of the crop in the parent image's pixel coordinates.
    """
    h, w = image_rgb.shape[:2]
    cx, cy = float(predicted_pixel_xy[0]), float(predicted_pixel_xy[1])

    if gt_pixel_xy is not None:
        gx, gy = float(gt_pixel_xy[0]), float(gt_pixel_xy[1])
        cx_center = 0.5 * (cx + gx)
        cy_center = 0.5 * (cy + gy)
        required = int(math.ceil(math.hypot(cx - gx, cy - gy) / 2)) + min_radius_px
    else:
        cx_center, cy_center = cx, cy
        required = min_radius_px

    radius = max(int(zoom_radius_px), required)
    x0 = max(0, int(round(cx_center - radius)))
    y0 = max(0, int(round(cy_center - radius)))
    x1 = min(w, int(round(cx_center + radius)))
    y1 = min(h, int(round(cy_center + radius)))
    return image_rgb[y0:y1, x0:x1].copy(), (x0, y0)


# ----------------------------------------------------------------------------
# High-level figure layouts
# ----------------------------------------------------------------------------


def render_localization_result(
    uav_image_path: str,
    satellite_image_path: str,
    predicted_pixel_xy: Tuple[float, float],
    gt_pixel_xy: Optional[Tuple[float, float]] = None,
    error_distance_m: Optional[float] = None,
    zoom_radius_px: int = 1500,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    preview_max_dim: int = 4096,
):
    """3-panel result figure with both GT and prediction markers.

    * Left  - full satellite image, both markers, connecting line, error label
    * Center- zoom around the prediction (and GT, if available)
    * Right - the UAV view

    Returns the matplotlib ``Figure`` (also written to ``output_path`` if given).
    """
    sat_full = draw_gt_and_prediction(
        satellite_image_path=satellite_image_path,
        predicted_pixel_xy=predicted_pixel_xy,
        gt_pixel_xy=gt_pixel_xy,
        error_distance_m=error_distance_m,
    )

    # Zoom panel computed BEFORE downsampling the full panel so we keep sat
    # resolution in the zoom view.
    zoom_rgb, (zoom_x0, zoom_y0) = crop_zoom_around(
        image_rgb=sat_full,
        predicted_pixel_xy=predicted_pixel_xy,
        gt_pixel_xy=gt_pixel_xy,
        zoom_radius_px=zoom_radius_px,
    )

    # Downscale the full-sat panel only for display.
    sat_h, sat_w = sat_full.shape[:2]
    scale = min(1.0, float(preview_max_dim) / float(max(sat_w, sat_h, 1)))
    if scale < 1.0:
        preview_w = max(1, int(round(sat_w * scale)))
        preview_h = max(1, int(round(sat_h * scale)))
        sat_preview = cv2.resize(sat_full, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    else:
        sat_preview = sat_full

    uav_bgr = cv2.imread(uav_image_path)
    if uav_bgr is None:
        raise FileNotFoundError(f"Could not read UAV image: {uav_image_path}")
    uav_rgb = cv2.cvtColor(uav_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.4, 1.0])

    ax_sat = fig.add_subplot(gs[0, 0])
    ax_sat.imshow(sat_preview)
    sat_title = f"Full satellite map\n{os.path.basename(satellite_image_path)} ({sat_w}×{sat_h})"
    ax_sat.set_title(sat_title, fontsize=11, pad=8)
    ax_sat.axis("off")

    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_zoom.imshow(zoom_rgb)
    zoom_title = "Zoom around prediction"
    if error_distance_m is not None:
        zoom_title += f"\nerror = {error_distance_m:.1f} m"
    ax_zoom.set_title(zoom_title, fontsize=11, pad=8)
    ax_zoom.axis("off")

    ax_uav = fig.add_subplot(gs[0, 2])
    ax_uav.imshow(uav_rgb)
    ax_uav.set_title(f"UAV view\n{os.path.basename(uav_image_path)}", fontsize=11, pad=8)
    ax_uav.axis("off")

    # Shared legend below the figure.
    pred_patch = plt.Line2D(
        [0], [0],
        marker="x", color="white", markerfacecolor="red",
        markeredgecolor="red", markeredgewidth=3, markersize=14, linestyle="None",
        label="Predicted position",
    )
    gt_patch = plt.Line2D(
        [0], [0],
        marker="*", color="white", markerfacecolor="magenta",
        markeredgecolor="magenta", markersize=18, linestyle="None",
        label="Ground truth",
    )
    line_patch = plt.Line2D(
        [0], [0], color=(1.0, 220 / 255, 0), linewidth=3, label="GT - prediction error",
    )
    handles = [pred_patch]
    if gt_pixel_xy is not None:
        handles += [gt_patch, line_patch]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ----------------------------------------------------------------------------
# Backward-compatible 2-panel API (kept for existing callers)
# ----------------------------------------------------------------------------


def draw_predicted_position(
    satellite_image_path: str,
    predicted_pixel_xy: Tuple[float, float],
    gt_pixel_xy: Optional[Tuple[float, float]] = None,
    marker_size: int = 300,
    marker_thickness: int = 12,
) -> np.ndarray:
    """Backward-compatible wrapper around :func:`draw_gt_and_prediction`."""
    return draw_gt_and_prediction(
        satellite_image_path=satellite_image_path,
        predicted_pixel_xy=predicted_pixel_xy,
        gt_pixel_xy=gt_pixel_xy,
        error_distance_m=None,
        marker_size=marker_size,
        marker_thickness=marker_thickness,
    )


def render_match_figure(
    uav_image_path: str,
    satellite_image_path: str,
    predicted_pixel_xy: Tuple[float, float],
    gt_pixel_xy: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    preview_max_dim: int = 4096,
):
    """Two-panel UAV + satellite figure (legacy)."""
    sat_rgb = draw_gt_and_prediction(
        satellite_image_path=satellite_image_path,
        predicted_pixel_xy=predicted_pixel_xy,
        gt_pixel_xy=gt_pixel_xy,
    )
    sat_h, sat_w = sat_rgb.shape[:2]
    scale = min(1.0, float(preview_max_dim) / float(max(sat_w, sat_h, 1)))
    if scale < 1.0:
        preview_w = max(1, int(round(sat_w * scale)))
        preview_h = max(1, int(round(sat_h * scale)))
        sat_rgb = cv2.resize(sat_rgb, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

    uav_bgr = cv2.imread(uav_image_path)
    if uav_bgr is None:
        raise FileNotFoundError(f"Could not read UAV image: {uav_image_path}")
    uav_rgb = cv2.cvtColor(uav_bgr, cv2.COLOR_BGR2RGB)

    fig, (ax_sat, ax_uav) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("white")

    ax_sat.imshow(sat_rgb)
    ax_sat.set_title(f"UAV Position: {os.path.basename(uav_image_path)}", fontsize=14, pad=8)
    ax_sat.axis("off")

    ax_uav.imshow(uav_rgb)
    ax_uav.set_title("UAV View", fontsize=12, pad=8)
    ax_uav.axis("off")

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def gt_pixel_for_image(
    satellite_image_path: str,
    bounds: dict,
    gt_lat: float,
    gt_lon: float,
) -> Tuple[int, int]:
    """Convert a ground-truth lat/lon to a pixel in ``satellite_image_path``."""
    with Image.open(satellite_image_path) as im:
        w, h = im.size
    return latlon_to_pixel(gt_lat, gt_lon, bounds, w, h)


def iter_match_overlay(
    sat_rgb: np.ndarray,
    centroids: Iterable[Tuple[float, float]],
    color=(255, 220, 0),
    radius: int = 6,
) -> np.ndarray:
    """Draw small dots at given centroids - used for the sanity-check overlay."""
    out = sat_rgb.copy()
    for cx, cy in centroids:
        cv2.circle(out, (int(round(cx)), int(round(cy))), radius, color, -1)
    return out
