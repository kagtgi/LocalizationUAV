"""Visualization helpers for query results.

Headline figures:

* :func:`draw_gt_and_prediction` - full satellite image with PRED red X and
  GT magenta star joined by a yellow line annotated with the error.
* :func:`draw_top_n_predictions` - full satellite image with the top-N
  candidate patches drawn as numbered, color-graded markers (rank 1 = bright
  red, rank N = cool purple), optionally including the GT marker.
* :func:`render_localization_result` - 3-panel layout (rank-1 view): full
  satellite | zoom | UAV view.
* :func:`render_top_n_result` - 3-panel layout (top-N view): full satellite
  | zoom around rank-1 | UAV view, all top-N positions overlaid.

``draw_predicted_position`` and ``render_match_figure`` remain as thin
wrappers for backward compatibility.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..io.bounds import latlon_to_pixel

if TYPE_CHECKING:
    from .query import PatchPrediction


Image.MAX_IMAGE_PIXELS = None


# ===== colors (RGB tuples used directly on the BGR-> RGB-converted image arrays) =====
COLOR_PRED = (255, 0, 0)        # red X for the rank-1 predicted position
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
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    pad = max(4, int(round(font_scale * 6)))
    chip_w = tw + 2 * pad
    chip_h = th + baseline + 2 * pad
    chip_x = max(0, x - chip_w // 2)
    chip_y = max(0, y - marker_size // 2 - chip_h - pad)
    cv2.rectangle(img_rgb, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), COLOR_LEGEND_BG, -1)
    cv2.rectangle(img_rgb, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), color, max(2, text_thickness))
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
    """Render the full satellite with prediction X + GT star + connector + error label."""
    img_bgr = cv2.imread(satellite_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read satellite image: {satellite_image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    auto_size, auto_thick, font_scale, text_thick = _scale_for_size(max(h, w))
    marker_size = int(marker_size or auto_size)
    marker_thickness = int(marker_thickness or auto_thick)

    pred_xy = (float(predicted_pixel_xy[0]), float(predicted_pixel_xy[1]))

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
                -1,
            )
            cv2.rectangle(
                img_rgb,
                (mid_x - tw // 2 - pad, mid_y - th - baseline - pad),
                (mid_x + tw // 2 + pad, mid_y + pad),
                COLOR_CONNECTOR,
                max(2, text_thick),
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

    _put_labelled_marker(
        img_rgb, pred_xy, cv2.MARKER_TILTED_CROSS, COLOR_PRED, "PRED",
        marker_size, marker_thickness, font_scale, text_thick,
    )
    if gt_pixel_xy is not None:
        _put_labelled_marker(
            img_rgb, gt_pixel_xy, cv2.MARKER_STAR, COLOR_GT, "GT",
            marker_size, marker_thickness, font_scale, text_thick,
        )
    return img_rgb


def _rank_color(rank: int, n: int, cmap_name: str = "plasma_r") -> Tuple[int, int, int]:
    """Map a rank (1..n) to a 0-255 RGB tuple via a matplotlib colormap.

    With ``plasma_r``, rank 1 = bright yellow/red (high attention), rank n
    = dark purple (low attention).
    """
    cmap = plt.get_cmap(cmap_name)
    t = (rank - 1) / max(n - 1, 1)
    rgba = cmap(t)
    return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


def draw_top_n_predictions(
    satellite_image_path: str,
    top_n: Sequence["PatchPrediction"],
    gt_pixel_xy: Optional[Tuple[float, float]] = None,
    cmap_name: str = "plasma_r",
    highlight_top_k: int = 5,
    radius_top1: Optional[int] = None,
    radius_min: Optional[int] = None,
) -> np.ndarray:
    """Render the full satellite with the top-N predicted patches.

    * Rank 1 is drawn largest, bright color, with a labeled chip ("1");
    * Ranks 2..``highlight_top_k`` are drawn medium-large, colored by rank,
      with their rank number drawn at the centroid;
    * Ranks > ``highlight_top_k`` are drawn smaller, with their rank number;
    * Lower-ranked markers are drawn first so higher-ranked markers stay on top.
    * Optional GT marker is drawn last, in magenta.
    """
    img_bgr = cv2.imread(satellite_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read satellite image: {satellite_image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    if not top_n:
        return img_rgb

    n = len(top_n)
    image_dim = max(h, w)
    # Marker sizes scale with image dimension.
    r_top = int(radius_top1) if radius_top1 is not None else max(40, image_dim // 200)
    r_min = int(radius_min) if radius_min is not None else max(18, image_dim // 600)
    r_top = max(r_top, r_min + 6)

    # Iterate from low rank to high rank so high-ranked markers are drawn last (on top).
    ranked = sorted(top_n, key=lambda p: -p.rank)
    for pred in ranked:
        rank = int(pred.rank)
        x, y = int(round(pred.pixel_xy[0])), int(round(pred.pixel_xy[1]))
        color = _rank_color(rank, n, cmap_name=cmap_name)

        # Radius interpolation: rank 1 -> r_top ; rank n -> r_min.
        t = (rank - 1) / max(n - 1, 1)
        radius = int(round(r_top * (1 - t) + r_min * t))

        # Filled circle outline + thin white border for contrast on busy satellites.
        cv2.circle(img_rgb, (x, y), radius + 3, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img_rgb, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img_rgb, (x, y), radius, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        # Rank text inside the marker (or just below if too small).
        label = str(rank)
        font_scale = max(0.5, radius / 24.0)
        thickness = max(1, int(round(font_scale * 1.8)))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_color = (255, 255, 255) if rank > max(1, highlight_top_k // 2) else (0, 0, 0)
        if radius >= max(20, tw // 2 + 6):
            tx = x - tw // 2
            ty = y + th // 2
        else:
            tx = x - tw // 2
            ty = y + radius + th + 4
            text_color = (0, 0, 0)
        # Black halo around text for readability.
        cv2.putText(img_rgb, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(img_rgb, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                    thickness, cv2.LINE_AA)

    # Re-draw the top-1 with a labeled chip so it stands out.
    top1 = min(top_n, key=lambda p: p.rank)
    _, auto_thick, font_scale, text_thick = _scale_for_size(image_dim)
    _put_labelled_marker(
        img_rgb,
        xy=top1.pixel_xy,
        marker_type=cv2.MARKER_TILTED_CROSS,
        color=COLOR_PRED,
        label=f"#1 (votes={top1.vote_count})",
        marker_size=max(80, image_dim // 100),
        marker_thickness=auto_thick,
        font_scale=font_scale,
        text_thickness=text_thick,
    )

    # GT on top.
    if gt_pixel_xy is not None:
        _put_labelled_marker(
            img_rgb,
            xy=gt_pixel_xy,
            marker_type=cv2.MARKER_STAR,
            color=COLOR_GT,
            label="GT",
            marker_size=max(100, image_dim // 80),
            marker_thickness=auto_thick,
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
    """Crop a window around the prediction (auto-expanded to include GT)."""
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
    """3-panel figure showing only the rank-1 prediction + GT."""
    sat_full = draw_gt_and_prediction(
        satellite_image_path=satellite_image_path,
        predicted_pixel_xy=predicted_pixel_xy,
        gt_pixel_xy=gt_pixel_xy,
        error_distance_m=error_distance_m,
    )
    zoom_rgb, _ = crop_zoom_around(
        image_rgb=sat_full,
        predicted_pixel_xy=predicted_pixel_xy,
        gt_pixel_xy=gt_pixel_xy,
        zoom_radius_px=zoom_radius_px,
    )
    return _three_panel_figure(
        sat_full=sat_full,
        zoom_rgb=zoom_rgb,
        uav_image_path=uav_image_path,
        satellite_image_path=satellite_image_path,
        title=title or _default_title(error_distance_m),
        output_path=output_path,
        preview_max_dim=preview_max_dim,
        legend_handles=_legend_handles(include_gt=gt_pixel_xy is not None, include_topn=False),
        center_panel_title=_zoom_title(error_distance_m, "Zoom around prediction"),
    )


def render_top_n_result(
    uav_image_path: str,
    satellite_image_path: str,
    top_n: Sequence["PatchPrediction"],
    gt_pixel_xy: Optional[Tuple[float, float]] = None,
    error_distance_m: Optional[float] = None,
    zoom_radius_px: int = 1500,
    highlight_top_k: int = 5,
    cmap_name: str = "plasma_r",
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    preview_max_dim: int = 4096,
):
    """3-panel figure with the full top-N overlay on the satellite + zoom + UAV."""
    if not top_n:
        raise ValueError("top_n must be a non-empty sequence of PatchPrediction")

    sat_full = draw_top_n_predictions(
        satellite_image_path=satellite_image_path,
        top_n=top_n,
        gt_pixel_xy=gt_pixel_xy,
        cmap_name=cmap_name,
        highlight_top_k=highlight_top_k,
    )
    top1 = min(top_n, key=lambda p: p.rank)
    zoom_rgb, _ = crop_zoom_around(
        image_rgb=sat_full,
        predicted_pixel_xy=top1.pixel_xy,
        gt_pixel_xy=gt_pixel_xy,
        zoom_radius_px=zoom_radius_px,
    )
    return _three_panel_figure(
        sat_full=sat_full,
        zoom_rgb=zoom_rgb,
        uav_image_path=uav_image_path,
        satellite_image_path=satellite_image_path,
        title=title or _default_title(error_distance_m, prefix=f"Top {len(top_n)} predictions"),
        output_path=output_path,
        preview_max_dim=preview_max_dim,
        legend_handles=_legend_handles(
            include_gt=gt_pixel_xy is not None,
            include_topn=True,
            n=len(top_n),
        ),
        center_panel_title=_zoom_title(error_distance_m, "Zoom around #1"),
    )


def _default_title(error_distance_m: Optional[float], prefix: str = "Localization result") -> str:
    if error_distance_m is not None:
        return f"{prefix}  (error vs GT = {error_distance_m:.1f} m)"
    return prefix


def _zoom_title(error_distance_m: Optional[float], base: str) -> str:
    if error_distance_m is not None:
        return f"{base}\nerror = {error_distance_m:.1f} m"
    return base


def _legend_handles(include_gt: bool, include_topn: bool, n: int = 0) -> List:
    handles = []
    if include_topn:
        handles.append(plt.Line2D(
            [0], [0], marker="o", color="white", markerfacecolor="#e76e5b",
            markeredgecolor="black", markeredgewidth=1.5, markersize=12, linestyle="None",
            label=f"Top {n} ranked candidates",
        ))
        handles.append(plt.Line2D(
            [0], [0], marker="x", color="white", markerfacecolor="red",
            markeredgecolor="red", markeredgewidth=3, markersize=14, linestyle="None",
            label="#1 plurality winner",
        ))
    else:
        handles.append(plt.Line2D(
            [0], [0], marker="x", color="white", markerfacecolor="red",
            markeredgecolor="red", markeredgewidth=3, markersize=14, linestyle="None",
            label="Predicted position",
        ))
    if include_gt:
        handles.append(plt.Line2D(
            [0], [0], marker="*", color="white", markerfacecolor="magenta",
            markeredgecolor="magenta", markersize=18, linestyle="None",
            label="Ground truth",
        ))
        if not include_topn:
            handles.append(plt.Line2D(
                [0], [0], color=(1.0, 220 / 255, 0), linewidth=3, label="GT - prediction error",
            ))
    return handles


def _three_panel_figure(
    sat_full: np.ndarray,
    zoom_rgb: np.ndarray,
    uav_image_path: str,
    satellite_image_path: str,
    title: Optional[str],
    output_path: Optional[str],
    preview_max_dim: int,
    legend_handles: List,
    center_panel_title: str,
):
    """Shared 3-panel layout: full sat | zoom | UAV view."""
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
    ax_sat.set_title(
        f"Full satellite map\n{os.path.basename(satellite_image_path)} ({sat_w}×{sat_h})",
        fontsize=11, pad=8,
    )
    ax_sat.axis("off")

    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_zoom.imshow(zoom_rgb)
    ax_zoom.set_title(center_panel_title, fontsize=11, pad=8)
    ax_zoom.axis("off")

    ax_uav = fig.add_subplot(gs[0, 2])
    ax_uav.imshow(uav_rgb)
    ax_uav.set_title(f"UAV view\n{os.path.basename(uav_image_path)}", fontsize=11, pad=8)
    ax_uav.axis("off")

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
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
# Backward-compatible 2-panel API
# ----------------------------------------------------------------------------


def draw_predicted_position(
    satellite_image_path: str,
    predicted_pixel_xy: Tuple[float, float],
    gt_pixel_xy: Optional[Tuple[float, float]] = None,
    marker_size: int = 300,
    marker_thickness: int = 12,
) -> np.ndarray:
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
    with Image.open(satellite_image_path) as im:
        w, h = im.size
    return latlon_to_pixel(gt_lat, gt_lon, bounds, w, h)


def iter_match_overlay(
    sat_rgb: np.ndarray,
    centroids: Iterable[Tuple[float, float]],
    color=(255, 220, 0),
    radius: int = 6,
) -> np.ndarray:
    out = sat_rgb.copy()
    for cx, cy in centroids:
        cv2.circle(out, (int(round(cx)), int(round(cy))), radius, color, -1)
    return out
