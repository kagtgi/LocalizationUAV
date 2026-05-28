import os
import sys

if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    project_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    __package__ = f"{project_name}.satellite"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import shutil
from PIL import Image
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from torchvision import transforms
from tqdm.auto import tqdm

from ..model import load_model
from ..dataset import process_uav
from ..geometry.ekeland import compute_expansion_ekeland_for_all_triangles
from ..io.export import export_expansion_ekeland_to_csv, save_drone_style_csv
from ..segmentation.contours import (
    contour_to_polygon,
    contour_to_polygon_dynamic,
    extract_contours_from_mask,
    filter_large_polygons,
    filter_large_polygons_dynamic,
    visualize_ekeland,
)

# === BÊN DƯỚI GIỮ NGUYÊN TOÀN BỘ CODE CŨ CỦA BẠN TỪ HÀM `compare_uav_satellite_segmentation` ===


def compare_uav_satellite_segmentation(
    uav_mask,
    satellite_mask,
    center_x,
    center_y,
    threshold=0.5,
    output_path=None,
    metric_mode="binary",
):
    """Crop satellite mask around UAV center and compare in binary or soft mode."""
    if uav_mask is None or satellite_mask is None:
        raise ValueError("uav_mask and satellite_mask must not be None")

    uav_mask_np = np.asarray(uav_mask)
    sat_mask_np = np.asarray(satellite_mask)
    metric_mode = str(metric_mode).strip().lower()
    if metric_mode not in {"binary", "soft"}:
        raise ValueError("metric_mode must be 'binary' or 'soft'")

    if uav_mask_np.ndim != 2 or sat_mask_np.ndim != 2:
        raise ValueError("uav_mask and satellite_mask must be 2D arrays")

    u_h, u_w = uav_mask_np.shape
    s_h, s_w = sat_mask_np.shape

    cx = int(np.clip(int(round(center_x)), 0, s_w - 1))
    cy = int(np.clip(int(round(center_y)), 0, s_h - 1))

    x0 = max(0, cx - u_w // 2)
    y0 = max(0, cy - u_h // 2)
    x1 = min(s_w, x0 + u_w)
    y1 = min(s_h, y0 + u_h)

    sat_crop = sat_mask_np[y0:y1, x0:x1]
    if sat_crop.shape != (u_h, u_w):
        sat_crop = cv2.resize(sat_crop.astype(np.float32), (u_w, u_h), interpolation=cv2.INTER_LINEAR)
    sat_crop = sat_crop.astype(np.float32)

    if metric_mode == "soft":
        threshold_source = "not_used"
        threshold_value = None
        uav_eval = np.clip(uav_mask_np.astype(np.float32), 0.0, 1.0)
        sat_eval = np.clip(sat_crop, 0.0, 1.0)
        inter = float(np.minimum(uav_eval, sat_eval).sum())
        union = float(np.maximum(uav_eval, sat_eval).sum())
        uav_pos = float(uav_eval.sum())
        sat_pos = float(sat_eval.sum())
    else:
        threshold_source = "fixed"
        if isinstance(threshold, str):
            if threshold.strip().lower() != "otsu":
                raise ValueError("threshold must be float or 'otsu'")
            crop_u8 = np.clip(sat_crop * 255.0, 0, 255).astype(np.uint8)
            otsu_thr_u8, _ = cv2.threshold(crop_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold_value = float(otsu_thr_u8) / 255.0
            threshold_source = "otsu"
        else:
            threshold_value = float(threshold)

        uav_eval = (uav_mask_np > threshold_value).astype(np.float32)
        sat_eval = (sat_crop > threshold_value).astype(np.float32)
        inter = float(np.logical_and(uav_eval == 1, sat_eval == 1).sum())
        union = float(np.logical_or(uav_eval == 1, sat_eval == 1).sum())
        uav_pos = float((uav_eval == 1).sum())
        sat_pos = float((sat_eval == 1).sum())

    iou = float(inter / union) if union > 0 else 0.0
    dice = float((2 * inter) / (uav_pos + sat_pos)) if (uav_pos + sat_pos) > 0 else 0.0
    precision = float(inter / sat_pos) if sat_pos > 0 else 0.0
    recall = float(inter / uav_pos) if uav_pos > 0 else 0.0

    metrics = {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "metric_mode": metric_mode,
        "uav_positive_pixels": uav_pos,
        "sat_crop_positive_pixels": sat_pos,
        "intersection_pixels": inter,
        "union_pixels": union,
        "uav_positive_mass": uav_pos,
        "sat_crop_positive_mass": sat_pos,
        "intersection_mass": inter,
        "union_mass": union,
        "crop_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "center_xy": [int(cx), int(cy)],
        "threshold": None if threshold_value is None else float(threshold_value),
        "threshold_source": threshold_source,
        "sat_crop_min": float(np.min(sat_crop)),
        "sat_crop_max": float(np.max(sat_crop)),
        "sat_crop_mean": float(np.mean(sat_crop)),
        "sat_crop_q95": float(np.quantile(sat_crop, 0.95)),
    }

    if output_path:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(uav_eval, cmap="magma" if metric_mode == "soft" else "gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("UAV soft mask" if metric_mode == "soft" else "UAV mask")
        axes[0].axis("off")

        axes[1].imshow(sat_eval if metric_mode == "binary" else sat_crop, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1].set_title(
            (
                "Satellite soft crop"
                if metric_mode == "soft"
                else f"Satellite binary crop\nthr={threshold_value:.3f}"
            )
            + f"\nmin={metrics['sat_crop_min']:.3f}, "
            f"max={metrics['sat_crop_max']:.3f}, q95={metrics['sat_crop_q95']:.3f}"
        )
        axes[1].axis("off")

        overlay = np.zeros((u_h, u_w, 3), dtype=np.float32)
        overlay[..., 0] = sat_eval
        overlay[..., 1] = uav_eval
        axes[2].imshow(overlay)
        if metric_mode == "soft":
            axes[2].set_title(f"Soft IoU={iou:.3f} | Dice={dice:.3f}")
        else:
            axes[2].set_title(f"IoU={iou:.3f} | Dice={dice:.3f} | thr={threshold_value:.3f}")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return metrics


def _compute_dynamic_patch_threshold(scores, base_threshold=0.5, quantile=0.6, min_threshold=0.35, max_threshold=0.85):
    """Compute dynamic per-patch score threshold from score distribution."""
    if scores is None or len(scores) == 0:
        return float(base_threshold)
    arr = scores.numpy() if hasattr(scores, "numpy") else np.asarray(scores)
    q = float(np.quantile(arr, quantile))
    th = max(float(base_threshold), q)
    return float(np.clip(th, min_threshold, max_threshold))


def _extract_polygon_cascade(mask, method='marching_squares', image_shape=None):
    """
    Cascade contour extraction:
    - small stage: low min_area + tiny epsilon
    - medium stage: default
    - large stage: high min_area + higher epsilon
    """
    if image_shape is None:
        image_shape = mask.shape

    cascade_cfg = [
        {"min_area": 20, "epsilon": (0.0015, 0.008)},
        {"min_area": 60, "epsilon": (0.0020, 0.015)},
        {"min_area": 200, "epsilon": (0.0040, 0.030)},
    ]

    contour_bucket = []
    polygon_bucket = []
    seen = set()

    for stage in cascade_cfg:
        contours = extract_contours_from_mask(mask, min_area=stage["min_area"], method=method)
        for contour in contours:
            polygon = contour_to_polygon_dynamic(
                contour,
                image_shape=image_shape,
                min_epsilon_factor=stage["epsilon"][0],
                max_epsilon_factor=stage["epsilon"][1],
            )
            if len(polygon) < 3:
                continue
            key = tuple((round(float(x), 1), round(float(y), 1)) for x, y in polygon)
            if key in seen:
                continue
            seen.add(key)
            contour_bucket.append(contour)
            polygon_bucket.append(polygon)

    return contour_bucket, polygon_bucket

def _normalize_scales_and_weights(scale, scales=None, scale_weights=None):
    """Normalize scales/weights while keeping backward-compatible `scale`."""
    if scales is None:
        scales = [float(scale)]
    else:
        scales = [float(s) for s in scales]
        if len(scales) == 0:
            scales = [float(scale)]

    if scale_weights is None or len(scale_weights) != len(scales):
        weights = np.ones(len(scales), dtype=np.float64)
    else:
        weights = np.asarray(scale_weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        if float(weights.sum()) <= 0:
            weights = np.ones(len(scales), dtype=np.float64)

    weights = weights / float(weights.sum())
    return scales, weights

def _morphological_cleanup(binary_mask, open_kernel_size=3, close_kernel_size=5, min_area=50):
    """Mask cleanup: open + close + remove tiny connected components."""
    mask_np = binary_mask.astype(np.uint8)

    if open_kernel_size > 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel_open)

    if close_kernel_size > 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < int(min_area):
            mask_np[labels == i] = 0

    return mask_np

def _polygon_soft_score(polygon, soft_mask):
    """Score polygon by average soft-mask confidence inside polygon."""
    if len(polygon) < 3:
        return 0.0
    h, w = soft_mask.shape[:2]
    poly_np = np.array(polygon, dtype=np.int32)
    poly_np[:, 0] = np.clip(poly_np[:, 0], 0, w - 1)
    poly_np[:, 1] = np.clip(poly_np[:, 1], 0, h - 1)
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(region, [poly_np], 1)
    pix = soft_mask[region == 1]
    if pix.size == 0:
        return 0.0
    return float(np.mean(pix))

def _polygon_iou(poly_a, poly_b):
    """IoU for two shapely polygons."""
    inter = poly_a.intersection(poly_b).area
    if inter <= 0:
        return 0.0
    union = poly_a.area + poly_b.area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _extract_valid_polygons_from_soft_mask(
    soft_mask,
    image_shape,
    use_cascade=True,
    contour_method='marching_squares',
    min_area=50,
    epsilon_factor=0.02,
    dynamic_large_filter=True,
    max_size_ratio=0.20,
    dynamic_large_quantile=0.995,
    fallback_max_size_ratio=0.95,
    enable_otsu_cleanup=True,
    morph_open_kernel=3,
    morph_close_kernel=5,
    min_component_area=40,
    enable_dual_path_fusion=True,
    iou_dedup_threshold=0.85,
):
    """Convert one soft mask into fused polygon candidates using the satellite-style pipeline."""
    soft_mask = np.asarray(soft_mask, dtype=np.float32)
    if soft_mask.ndim != 2:
        raise ValueError("soft_mask must be a 2D array")

    if len(image_shape) == 3:
        image_shape_hw = image_shape[:2]
    else:
        image_shape_hw = image_shape

    # Repo path: soft-mask -> cascade polygons.
    if use_cascade:
        contours_a, polygons_a = _extract_polygon_cascade(
            mask=soft_mask,
            method=contour_method,
            image_shape=image_shape_hw,
        )
    else:
        contours_a = extract_contours_from_mask(soft_mask, min_area=min_area, method=contour_method)
        polygons_a = [
            contour_to_polygon_dynamic(
                c,
                image_shape=image_shape_hw,
                min_epsilon_factor=max(0.0005, epsilon_factor * 0.25),
                max_epsilon_factor=epsilon_factor,
            )
            for c in contours_a
        ]

    if dynamic_large_filter:
        contours_a, polygons_a = filter_large_polygons_dynamic(
            contours_a,
            polygons_a,
            image_shape_hw,
            quantile=dynamic_large_quantile,
            fallback_max_size_ratio=fallback_max_size_ratio,
        )
    else:
        contours_a, polygons_a = filter_large_polygons(
            contours_a,
            polygons_a,
            image_shape_hw,
            max_size_ratio=max_size_ratio,
        )

    # Factory path: Otsu + morphology + OpenCV contour.
    binary_mask = (soft_mask > 0.5).astype(np.uint8)
    contours_b = []
    polygons_b = []
    if enable_otsu_cleanup:
        mask_u8 = np.clip(soft_mask * 255.0, 0, 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_mask = _morphological_cleanup(
            binary_mask=binary_mask,
            open_kernel_size=morph_open_kernel,
            close_kernel_size=morph_close_kernel,
            min_area=min_component_area,
        )
        contours_b = extract_contours_from_mask(binary_mask, min_area=min_area, method='opencv')
        polygons_b = [contour_to_polygon(c, epsilon_factor=epsilon_factor) for c in contours_b]

        if dynamic_large_filter:
            contours_b, polygons_b = filter_large_polygons_dynamic(
                contours_b,
                polygons_b,
                image_shape_hw,
                quantile=dynamic_large_quantile,
                fallback_max_size_ratio=fallback_max_size_ratio,
            )
        else:
            contours_b, polygons_b = filter_large_polygons(
                contours_b,
                polygons_b,
                image_shape_hw,
                max_size_ratio=max_size_ratio,
            )

    candidates = []
    for contour, polygon in zip(contours_a, polygons_a):
        candidates.append(("repo", contour, polygon, _polygon_soft_score(polygon, soft_mask)))
    if enable_dual_path_fusion:
        for contour, polygon in zip(contours_b, polygons_b):
            candidates.append(("factory", contour, polygon, _polygon_soft_score(polygon, soft_mask)))

    valid_contours = []
    valid_polygons = []
    accepted_shapes = []
    accepted_sources = []

    if candidates:
        candidates.sort(key=lambda x: x[3], reverse=True)
        for source, contour, polygon, _score in candidates:
            if len(polygon) < 3:
                continue
            try:
                shp = Polygon(polygon)
                if (not shp.is_valid) or shp.area <= 1.0:
                    continue
            except Exception:
                continue

            is_dup = False
            for other in accepted_shapes:
                if _polygon_iou(shp, other) >= float(iou_dedup_threshold):
                    is_dup = True
                    break
            if is_dup:
                continue

            valid_contours.append(contour)
            valid_polygons.append(polygon)
            accepted_shapes.append(shp)
            accepted_sources.append(source)

    if accepted_sources:
        repo_n = sum(1 for s in accepted_sources if s == "repo")
        factory_n = sum(1 for s in accepted_sources if s == "factory")
        print(f"   Polygon fusion kept {len(valid_polygons)} | repo={repo_n}, factory={factory_n}")

    return {
        "binary_mask": binary_mask,
        "repo_contours": contours_a,
        "repo_polygons": polygons_a,
        "factory_contours": contours_b,
        "factory_polygons": polygons_b,
        "valid_contours": valid_contours,
        "valid_polygons": valid_polygons,
        "accepted_sources": accepted_sources,
    }

def create_blend_weight(patch_size):
    """Tạo ma trận trọng số để ghép ảnh mượt hơn (Pyramid/Hat shape)"""
    # Tạo tensor 1D từ 0 đến 1 rồi giảm về 0
    x = torch.linspace(0, 1, patch_size)
    y = torch.linspace(0, 1, patch_size)
    
    # Dạng kim tự tháp: cao ở giữa, thấp ở biên
    weight_x = 1 - torch.abs(2 * x - 1)
    weight_y = 1 - torch.abs(2 * y - 1)
    
    # Nhân outer product để tạo 2D grid
    weight = weight_x.view(1, -1) * weight_y.view(-1, 1)
    
    # Tránh chia cho 0
    return torch.clamp(weight, min=0.001)

def segment_at_scale(
    image,
    scale,
    model,
    device,
    patch_size,
    overlap,
    seg_threshold,
    uncertainty_ratio,
    dynamic_threshold=True,
    dynamic_quantile=0.6,
    min_seg_threshold=0.35,
    max_seg_threshold=0.85,
    use_uncertainty_filter=True,
):
    """Segment image at a specific scale"""
    # Resize image
    orig_w, orig_h = image.size
    scaled_w, scaled_h = int(orig_w * scale), int(orig_h * scale)
    scaled_img = image.resize((scaled_w, scaled_h), Image.BILINEAR)

    # Pad
    pad_w = (patch_size - scaled_w % patch_size) % patch_size
    pad_h = (patch_size - scaled_h % patch_size) % patch_size
    padded_w, padded_h = scaled_w + pad_w, scaled_h + pad_h

    if pad_w or pad_h:
        padded_img = Image.new("RGB", (padded_w, padded_h))
        padded_img.paste(scaled_img, (0, 0))
    else:
        padded_img = scaled_img

    # Setup
    transform = transforms.ToTensor()
    full_mask_sum = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    full_weight_sum = torch.zeros((padded_h, padded_w), dtype=torch.float32)

    stride = patch_size - overlap
    patch_coords = [(t, l) for t in range(0, padded_h, stride) for l in range(0, padded_w, stride)]

    # Blending weight
    weight = create_blend_weight(patch_size)

    # Segment
    for top, left in tqdm(patch_coords, desc=f"Scale {scale:.2f}x", leave=False):
        top = min(top, padded_h - patch_size)
        left = min(left, padded_w - patch_size)

        patch = padded_img.crop((left, top, left + patch_size, top + patch_size))
        patch_tensor = transform(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(patch_tensor)

        pred = predictions[0]
        combined_mask = torch.zeros((patch_size, patch_size), dtype=torch.float32)

        if 'masks' in pred and len(pred['masks']) > 0:
            scores = pred['scores'].cpu()
            masks = pred['masks'].cpu()

            patch_threshold = float(seg_threshold)
            if dynamic_threshold:
                patch_threshold = _compute_dynamic_patch_threshold(
                    scores=scores,
                    base_threshold=seg_threshold,
                    quantile=dynamic_quantile,
                    min_threshold=min_seg_threshold,
                    max_threshold=max_seg_threshold,
                )

            for i in range(masks.shape[0]):
                if scores[i] >= patch_threshold:
                    m = masks[i, 0]
                    if (not use_uncertainty_filter) or ((m > 0.6).float().mean().item() <= uncertainty_ratio):
                        combined_mask = torch.maximum(combined_mask, m)

        full_mask_sum[top:top+patch_size, left:left+patch_size] += combined_mask * weight
        full_weight_sum[top:top+patch_size, left:left+patch_size] += weight

    # Finalize
    full_mask = (full_mask_sum / (full_weight_sum + 1e-6)).clamp(0, 1)
    mask_at_scale = full_mask[:scaled_h, :scaled_w]

    # Resize back to original size
    mask_np = mask_at_scale.numpy()
    mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return mask_resized


def generate_satellite_soft_mask(
    sat_img_path,
    model,
    device,
    scale=1.0,
    patch_size=500,
    overlap=100,
    dynamic_threshold=True,
    dynamic_quantile=0.6,
    min_seg_threshold=0.35,
    max_seg_threshold=0.85,
    use_uncertainty_filter=False,
    uncertainty_ratio=0.5,
    output_npy_path=None,
):
    """Generate full-size soft mask (float [0,1]) for one satellite image."""
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(sat_img_path).convert("RGB")
    soft_mask = segment_at_scale(
        image=image,
        scale=scale,
        model=model,
        device=device,
        patch_size=patch_size,
        overlap=overlap,
        seg_threshold=0.5,
        uncertainty_ratio=uncertainty_ratio,
        dynamic_threshold=dynamic_threshold,
        dynamic_quantile=dynamic_quantile,
        min_seg_threshold=min_seg_threshold,
        max_seg_threshold=max_seg_threshold,
        use_uncertainty_filter=use_uncertainty_filter,
    )
    if output_npy_path:
        np.save(output_npy_path, soft_mask.astype(np.float32))
    return soft_mask


def generate_satellite_soft_mask_crop(
    sat_img_path,
    crop_bbox_xyxy,
    model,
    device,
    scale=1.0,
    patch_size=500,
    overlap=100,
    dynamic_threshold=True,
    dynamic_quantile=0.6,
    min_seg_threshold=0.35,
    max_seg_threshold=0.85,
    use_uncertainty_filter=False,
    uncertainty_ratio=0.5,
):
    """Generate soft mask only for a local satellite crop to avoid full-image OOM."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(sat_img_path) as image:
        img_w, img_h = image.size
        x0, y0, x1, y1 = [int(v) for v in crop_bbox_xyxy]
        x0 = max(0, min(x0, img_w - 1))
        y0 = max(0, min(y0, img_h - 1))
        x1 = max(x0 + 1, min(x1, img_w))
        y1 = max(y0 + 1, min(y1, img_h))

        crop_img = image.crop((x0, y0, x1, y1)).convert("RGB")

    soft_mask = segment_at_scale(
        image=crop_img,
        scale=scale,
        model=model,
        device=device,
        patch_size=patch_size,
        overlap=overlap,
        seg_threshold=0.5,
        uncertainty_ratio=uncertainty_ratio,
        dynamic_threshold=dynamic_threshold,
        dynamic_quantile=dynamic_quantile,
        min_seg_threshold=min_seg_threshold,
        max_seg_threshold=max_seg_threshold,
        use_uncertainty_filter=use_uncertainty_filter,
    )
    return soft_mask

def process_satellite_as_drone(
    tif_path,
    output_dir,
    model,
    device,
    scale=1.0,
    scales=None,
    scale_weights=None,
    patch_size=500,
    overlap=100,
    dynamic_threshold=True,
    dynamic_quantile=0.6,
    min_seg_threshold=0.35,
    max_seg_threshold=0.85,
    use_uncertainty_filter=False,
    uncertainty_ratio=0.5,
    use_cascade=True,
    contour_method='marching_squares',
    min_area=50,
    epsilon_factor=0.02,
    dynamic_large_filter=True,
    max_size_ratio=0.20,
    dynamic_large_quantile=0.995,
    fallback_max_size_ratio=0.95,
    max_search_radius=100.0,
    enable_otsu_cleanup=True,
    morph_open_kernel=3,
    morph_close_kernel=5,
    min_component_area=40,
    enable_dual_path_fusion=True,
    iou_dedup_threshold=0.85,
):
    """
    Process satellite TIFF as drone-like geometric feature extraction.
    Hybrid mode:
    1) multi-scale weighted soft-mask combine,
    2) repo cascade polygon path,
    3) optional Otsu+morphology polygon path,
    4) dual-path polygon fusion, then Delaunay + Ekeland + CSV export.
    """
    filename = os.path.basename(tif_path)
    base_name = os.path.splitext(filename)[0]
    csv_name = f"{base_name}.csv"
    output_csv = os.path.join(output_dir, csv_name)

    print(f"\nPROCESSING: {filename}")
    print(f"   Scale: {scale} | Output: {output_csv}")

    try:
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(tif_path).convert("RGB")

        # 1) Multi-scale segmentation + weighted combine
        scales_norm, weights = _normalize_scales_and_weights(
            scale=scale,
            scales=scales,
            scale_weights=scale_weights,
        )
        print(f"   Multi-scale: {scales_norm} | weights: {[round(float(w), 3) for w in weights]}")

        all_masks = []
        for sc in scales_norm:
            mask_sc = segment_at_scale(
                image=image,
                scale=sc,
                model=model,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
                seg_threshold=0.5,
                uncertainty_ratio=uncertainty_ratio,
                dynamic_threshold=dynamic_threshold,
                dynamic_quantile=dynamic_quantile,
                min_seg_threshold=min_seg_threshold,
                max_seg_threshold=max_seg_threshold,
                use_uncertainty_filter=use_uncertainty_filter,
            )
            all_masks.append(mask_sc.astype(np.float32))

        if len(all_masks) == 1:
            combined_mask = all_masks[0]
        else:
            combined_mask = np.average(np.stack(all_masks, axis=0), axis=0, weights=weights)

        polygon_data = _extract_valid_polygons_from_soft_mask(
            soft_mask=combined_mask,
            image_shape=(image.size[1], image.size[0]),
            use_cascade=use_cascade,
            contour_method=contour_method,
            min_area=min_area,
            epsilon_factor=epsilon_factor,
            dynamic_large_filter=dynamic_large_filter,
            max_size_ratio=max_size_ratio,
            dynamic_large_quantile=dynamic_large_quantile,
            fallback_max_size_ratio=fallback_max_size_ratio,
            enable_otsu_cleanup=enable_otsu_cleanup,
            morph_open_kernel=morph_open_kernel,
            morph_close_kernel=morph_close_kernel,
            min_component_area=min_component_area,
            enable_dual_path_fusion=enable_dual_path_fusion,
            iou_dedup_threshold=iou_dedup_threshold,
        )
        valid_polygons = polygon_data["valid_polygons"]

        if not valid_polygons:
            print("   No valid polygons found.")
            return

        # 3) Delaunay + Ekeland
        all_obstacles = [Polygon(p) for p in valid_polygons if len(p) >= 3]
        all_expansion_results = []

        print(f"   Computing Ekeland for {len(valid_polygons)} polygons...")
        for poly_idx, polygon in enumerate(tqdm(valid_polygons, desc="Polygons")):
            if len(polygon) < 3:
                continue

            try:
                points = np.array(polygon, dtype=np.float64)
                tri = Delaunay(points)

                current_obstacles = [obs for i, obs in enumerate(all_obstacles) if i != poly_idx]

                expansion_res = compute_expansion_ekeland_for_all_triangles(
                    tri,
                    all_polygons_in_image=current_obstacles,
                    max_search_radius=max_search_radius,
                )

                all_expansion_results.extend(expansion_res)
            except Exception:
                continue

        # 4) Export CSV (schema unchanged)
        save_drone_style_csv(all_expansion_results, output_csv)

    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()


def segment_and_save_satellite_mask(
    tif_path,
    output_dir,
    model,
    device,
    scale=1.0,
    scales=None,
    scale_weights=None,
    patch_size=500,
    overlap=100,
    dynamic_threshold=True,
    dynamic_quantile=0.65,
    min_seg_threshold=0.30,
    max_seg_threshold=0.80,
    use_uncertainty_filter=True,
    uncertainty_ratio=0.60,
    enable_otsu_cleanup=True,
    morph_open_kernel=3,
    morph_close_kernel=5,
    min_component_area=40,
    save_soft_mask_npy=False
):
    """
    Hàm thực hiện segmentation toàn bộ ảnh vệ tinh bằng multi-scale,
    áp dụng cleanup và lưu kết quả dưới dạng ảnh mask trắng đen (.png).
    Không thực hiện trích xuất Ekeland features.
    """
    filename = os.path.basename(tif_path)
    base_name = os.path.splitext(filename)[0]

    # Khởi tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    out_img_path = os.path.join(output_dir, f"{base_name}_segmentation_mask.png")
    
    print(f"\n[SEGMENTATION ONLY] PROCESSING: {filename}")

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(tif_path).convert("RGB")

    # 1. Chuẩn hóa scales và weights
    scales_norm, weights = _normalize_scales_and_weights(
        scale=scale,
        scales=scales,
        scale_weights=scale_weights,
    )
    print(f"   Multi-scale: {scales_norm} | weights: {[round(float(w), 3) for w in weights]}")

    # 2. Chạy segmentation theo từng scale
    all_masks = []
    for sc in scales_norm:
        mask_sc = segment_at_scale(
            image=image,
            scale=sc,
            model=model,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            seg_threshold=0.5, # Threshold cơ sở, sẽ được ghi đè nếu dynamic_threshold=True
            uncertainty_ratio=uncertainty_ratio,
            dynamic_threshold=dynamic_threshold,
            dynamic_quantile=dynamic_quantile,
            min_seg_threshold=min_seg_threshold,
            max_seg_threshold=max_seg_threshold,
            use_uncertainty_filter=use_uncertainty_filter,
        )
        all_masks.append(mask_sc.astype(np.float32))

    # 3. Kết hợp các mask theo trọng số (Soft mask [0, 1])
    if len(all_masks) == 1:
        combined_mask = all_masks[0]
    else:
        combined_mask = np.average(np.stack(all_masks, axis=0), axis=0, weights=weights)

    # (Tùy chọn) Lưu soft mask dạng .npy để dùng cho việc visualize heatmap sau này
    if save_soft_mask_npy:
        out_npy_path = os.path.join(output_dir, f"{base_name}_soft_mask.npy")
        np.save(out_npy_path, combined_mask)
        print(f"   ✓ Đã lưu soft mask raw tại: {out_npy_path}")

    # 4. Hậu xử lý (Otsu & Morphology) để tạo ra Binary Mask rõ nét
    if enable_otsu_cleanup:
        print("   Đang áp dụng Otsu Thresholding và Morphological Cleanup...")
        mask_u8 = np.clip(combined_mask * 255.0, 0, 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        binary_mask = _morphological_cleanup(
            binary_mask=binary_mask,
            open_kernel_size=morph_open_kernel,
            close_kernel_size=morph_close_kernel,
            min_area=min_component_area,
        )
        # Chuyển đổi [0, 1] về [0, 255] để lưu ảnh
        final_vis_mask = binary_mask * 255
    else:
        # Nếu không dùng Otsu, lưu lại kết quả soft mask được ánh xạ lên thang 255
        final_vis_mask = np.clip(combined_mask * 255.0, 0, 255).astype(np.uint8)

    # 5. Lưu kết quả
    cv2.imwrite(out_img_path, final_vis_mask)
    print(f"   ✓ Đã lưu mask ảnh vệ tinh (.png) tại: {out_img_path}")

    return final_vis_mask

# ==============================================================================
# complete_segmentation_demo_uav_with_rotation()
# ==============================================================================

def complete_segmentation_demo_uav_with_rotation(
    model, 
    device, 
    uav_img_path, 
    uav_csv_path,
    method='marching_squares', 
    max_search_radius=100.0,
    return_details=False,
    output_dir=None, 
    max_vis_ekeland_triangles=None
):

    print("=" * 70)
    print("UAV EKELAND EXTRACTION (North-aligned)")
    print("=" * 70)

    print("\n[INFO] Step 1: Rotating UAV image to North-up...")
    try:
        img_raw, img_rotated, meta = process_uav(
            img_path=uav_img_path,
            csv_path=uav_csv_path
        )
        print(f"[INFO] Rotated: yaw={meta['Yaw (Phi)']:.1f}°, "
              f"roll={meta['Roll (Kappa)']:.1f}°, pitch={meta['Pitch (Omega)']:.1f}°")
        print(f"[INFO] Rotated image size: {img_rotated.size}")
    except Exception as e:
        print(f"[ERROR] Error rotating image: {e}")
        print(f"[WARNING] Fallback: Using original image (WARNING: Ekeland may not be accurate!)")
        img_raw = Image.open(uav_img_path).convert("RGB")
        img_rotated = img_raw.resize((500, 500))
        meta = {"Yaw (Phi)": 0.0}

    print("\n[INFO] Step 2: Segmentation on the rotated image...")
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(img_rotated).to(device)
    img_np = np.array(img_rotated)

    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    scores = pred['scores'].cpu().numpy()
    keep = scores >= 0.5

    if not keep.any():
        print("[INFO] No predictions have sufficient confidence.")
        if return_details:
            return {
                "csv_path": None,
                "uav_mask": None,
                "uav_soft_mask": None,
                "meta": meta,
                "rotated_size": img_rotated.size,
            }
        return None

    pred_masks = pred['masks'][keep].cpu().numpy()
    pred_scores = pred['scores'][keep].cpu().numpy()
    print(f"[INFO] {len(pred_masks)} building segmented")

    print("\n[INFO] Step 3: Extracting Ekeland features...")

    combined_soft_mask = np.zeros(pred_masks[0][0].shape, dtype=np.float32)
    for mask, score in zip(pred_masks, pred_scores):
        binary_mask = mask[0] if len(mask.shape) == 3 else mask
        combined_soft_mask = np.maximum(combined_soft_mask, binary_mask.astype(np.float32))

    polygon_data = _extract_valid_polygons_from_soft_mask(
        soft_mask=combined_soft_mask,
        image_shape=img_np.shape,
        use_cascade=True,
        contour_method=method,
        min_area=50,
        epsilon_factor=0.02,
        dynamic_large_filter=True,
        max_size_ratio=0.85,
        dynamic_large_quantile=0.995,
        fallback_max_size_ratio=0.90,
        enable_otsu_cleanup=True,
        morph_open_kernel=3,
        morph_close_kernel=5,
        min_component_area=50,
        enable_dual_path_fusion=True,
        iou_dedup_threshold=0.85,
    )
    combined_contours = polygon_data["valid_contours"]
    combined_polygons = polygon_data["valid_polygons"]
    combined_mask = polygon_data["binary_mask"]

    if not combined_polygons:
        print("[INFO] No valid polygons found.")
        if return_details:
            return {
                "csv_path": None,
                "uav_mask": np.array(combined_mask, dtype=np.uint8),
                "uav_soft_mask": combined_soft_mask.astype(np.float32),
                "meta": meta,
                "rotated_size": img_rotated.size,
            }
        return None

    print(f"[INFO] {len(combined_polygons)} available polygons")

    all_obstacle_polygons = []
    for poly in combined_polygons:
        try:
            sp = Polygon(poly)
            if sp.is_valid:
                all_obstacle_polygons.append(sp)
        except:
            pass

    all_expansion_results = []
    polygon_triangulations = []

    for poly_idx, polygon in enumerate(combined_polygons):
        if len(polygon) < 3:
            polygon_triangulations.append(None)
            continue
        try:
            points = np.array(polygon, dtype=np.float64)
            tri = Delaunay(points)
            polygon_triangulations.append({
                'points': points, 'triangles': tri.simplices, 'tri_object': tri
            })
            other_obstacles = [obs for j, obs in enumerate(all_obstacle_polygons) if j != poly_idx]
            exp_results = compute_expansion_ekeland_for_all_triangles(
                tri, all_polygons_in_image=other_obstacles, max_search_radius=max_search_radius
            )
            all_expansion_results.extend(exp_results)
        except Exception as e:
            print(f"    Lỗi polygon {poly_idx}: {e}")
            polygon_triangulations.append(None)

    base_name = os.path.splitext(os.path.basename(uav_img_path))[0]
    csv_filename = f"ekeland_{base_name}_rotated.csv"
    
    if output_dir:
        csv_filepath = os.path.join(output_dir, csv_filename)
    else:
        csv_filepath = csv_filename

    if all_expansion_results:
        export_expansion_ekeland_to_csv(all_expansion_results, csv_filepath)
        print(f"\n[INFO] CSV fetures saved at: {csv_filepath}")
        print(f"[INFO] This CSV was extracted from the rotated image -> can be used for matching!")
    else:
        print("[INFO] No results to export to CSV")
        csv_filepath = None

    print("\n[INFO] Step 4: Visualizing Ekeland results...")
    if combined_contours and combined_polygons:
        combined_mask_vis = (np.array(combined_mask, dtype=np.uint8)) * 255
        visualize_ekeland(
            img_np, combined_mask_vis, combined_contours, combined_polygons,
            polygon_triangulations, all_expansion_results,
            f"UAV Ekeland (North-up, yaw={meta['Yaw (Phi)']:.1f}° corrected)",
            max_ekeland_triangles=max_vis_ekeland_triangles  
        )

        default_save_name = "UAV_6_panels_visualization.jpg"
        if os.path.exists(default_save_name):
            if output_dir:
                vis_path = os.path.join(output_dir, f"6_panels_{base_name}.jpg")
            else:
                vis_path = f"6_panels_{base_name}.jpg"
                
            shutil.move(default_save_name, vis_path)
            print(f"[INFO] Saved 6-step image at: {vis_path}")

    if return_details:
        return {
            "csv_path": csv_filepath,
            "uav_mask": np.array(combined_mask, dtype=np.uint8),
            "uav_soft_mask": combined_soft_mask.astype(np.float32),
            "meta": meta,
            "rotated_size": img_rotated.size,
        }
    return csv_filepath

def complete_segmentation_demo_uav_with_rotation_multiscale(
    model, device, uav_img_path, uav_csv_path,
    method='marching_squares', max_search_radius=100.0,
    return_details=False,
    output_dir=None, # ĐÃ THÊM THAM SỐ OUTPUT_DIR ĐỂ LƯU ẢNH
    scales=[0.75], 
    scale_weights=[1.0], 
    patch_size=500,
    overlap=100,
    dynamic_threshold=True,
    dynamic_quantile=0.6,
    min_seg_threshold=0.35,
    max_seg_threshold=0.85,
    use_uncertainty_filter=True,
    uncertainty_ratio=0.5,
    enable_otsu_cleanup=True,
    morph_open_kernel=3,
    morph_close_kernel=5,
    min_component_area=50,
    soft_mask_scale_for_compare=1.0,
):
    print("=" * 70)
    print("UAV EKELAND EXTRACTION (North-aligned + Multi-scale)")
    print("=" * 70)

    print("\nBước 1: Xoay ảnh UAV về North-up...")
    try:
        img_raw, img_rotated, meta = process_uav(img_path=uav_img_path, csv_path=uav_csv_path)
        print(f"  ✓ Đã xoay: yaw={meta['Yaw (Phi)']:.1f}°, kích thước: {img_rotated.size}")
    except Exception as e:
        print(f"  ✗ Lỗi xoay ảnh: {e}")
        return None

    img_np = np.array(img_rotated)

    print("\nBước 2: Segmentation đa tỷ lệ trên ảnh UAV đã xoay...")
    model.eval()

    scales_norm, weights = _normalize_scales_and_weights(scale=1.0, scales=scales, scale_weights=scale_weights)
    all_masks = []
    compare_scale = float(soft_mask_scale_for_compare)
    compare_soft_mask = None
    
    for sc in scales_norm:
        mask_sc = segment_at_scale(
            image=img_rotated,
            scale=sc,
            model=model,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            seg_threshold=0.5,
            uncertainty_ratio=uncertainty_ratio,
            dynamic_threshold=dynamic_threshold,
            dynamic_quantile=dynamic_quantile,
            min_seg_threshold=min_seg_threshold,
            max_seg_threshold=max_seg_threshold,
            use_uncertainty_filter=use_uncertainty_filter,
        )
        mask_sc = mask_sc.astype(np.float32)
        all_masks.append(mask_sc)
        if compare_soft_mask is None and abs(float(sc) - compare_scale) < 1e-8:
            compare_soft_mask = mask_sc

    if len(all_masks) == 1:
        combined_soft_mask = all_masks[0]
    else:
        combined_soft_mask = np.average(np.stack(all_masks, axis=0), axis=0, weights=weights)

    if compare_soft_mask is None:
        compare_soft_mask = segment_at_scale(
            image=img_rotated,
            scale=compare_scale,
            model=model,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            seg_threshold=0.5,
            uncertainty_ratio=uncertainty_ratio,
            dynamic_threshold=dynamic_threshold,
            dynamic_quantile=dynamic_quantile,
            min_seg_threshold=min_seg_threshold,
            max_seg_threshold=max_seg_threshold,
            use_uncertainty_filter=use_uncertainty_filter,
        ).astype(np.float32)

    print(f"[INFO] UAV compare soft mask scale: {compare_scale:.2f}x")

    print(f"[INFO] Step 3: Extracting stronger polygons (cascade + Otsu + fusion)...")
    polygon_data = _extract_valid_polygons_from_soft_mask(
        soft_mask=combined_soft_mask,
        image_shape=img_np.shape,
        use_cascade=True,
        contour_method=method,
        min_area=min_component_area,
        epsilon_factor=0.02,
        dynamic_large_filter=True,
        max_size_ratio=0.85,
        dynamic_large_quantile=0.995,
        fallback_max_size_ratio=0.90,
        enable_otsu_cleanup=enable_otsu_cleanup,
        morph_open_kernel=morph_open_kernel,
        morph_close_kernel=morph_close_kernel,
        min_component_area=min_component_area,
        enable_dual_path_fusion=True,
        iou_dedup_threshold=0.85,
    )
    binary_mask = polygon_data["binary_mask"]
    combined_contours = polygon_data["valid_contours"]
    combined_polygons = polygon_data["valid_polygons"]

    if not combined_polygons:
        print("  Không có polygon hợp lệ.")
        if return_details:
            return {
                "csv_path": None,
                "uav_mask": binary_mask,
                "uav_soft_mask": compare_soft_mask.astype(np.float32),
                "meta": meta,
                "rotated_size": img_rotated.size,
            }
        return None

    print(f"[INFO] {len(combined_polygons)} available polygons to compute Ekeland")

    all_obstacle_polygons = []
    for poly in combined_polygons:
        try:
            sp = Polygon(poly)
            if sp.is_valid:
                all_obstacle_polygons.append(sp)
        except: pass

    all_expansion_results = []
    polygon_triangulations = []

    for poly_idx, polygon in enumerate(combined_polygons):
        if len(polygon) < 3:
            polygon_triangulations.append(None)
            continue
        try:
            points = np.array(polygon, dtype=np.float64)
            tri = Delaunay(points)
            polygon_triangulations.append({'points': points, 'triangles': tri.simplices, 'tri_object': tri})
            other_obstacles = [obs for j, obs in enumerate(all_obstacle_polygons) if j != poly_idx]
            exp_results = compute_expansion_ekeland_for_all_triangles(
                tri, all_polygons_in_image=other_obstacles, max_search_radius=max_search_radius
            )
            all_expansion_results.extend(exp_results)
        except Exception:
            polygon_triangulations.append(None)

    base_name = os.path.splitext(os.path.basename(uav_img_path))[0]
    csv_filename = f"ekeland_{base_name}_rotated.csv"
    
    if output_dir:
        csv_filepath = os.path.join(output_dir, csv_filename)
    else:
        csv_filepath = csv_filename

    if all_expansion_results:
        export_expansion_ekeland_to_csv(all_expansion_results, csv_filepath)
        print(f"\n[INFO] CSV saved at: {csv_filepath}")
    else:
        csv_filepath = None

    if combined_contours and combined_polygons:
        combined_mask_vis = (np.array(binary_mask, dtype=np.uint8)) * 255
        visualize_ekeland(
            img_np, combined_mask_vis, combined_contours, combined_polygons,
            polygon_triangulations, all_expansion_results,
            f"UAV Ekeland (North-up, yaw={meta['Yaw (Phi)']:.1f}° corrected)"
        )
        
        default_save_name = "UAV_6_panels_visualization.jpg"
        if os.path.exists(default_save_name):
            if output_dir:
                vis_path = os.path.join(output_dir, f"6_panels_{base_name}.jpg")
            else:
                vis_path = f"6_panels_{base_name}.jpg"
            shutil.move(default_save_name, vis_path)
            print(f"  ✓ Đã lưu ảnh 6 bước tại: {vis_path}")


    if return_details:
        return {
            "csv_path": csv_filepath,
            "uav_mask": binary_mask, 
            "uav_soft_mask": compare_soft_mask.astype(np.float32),
            "meta": meta,
            "rotated_size": img_rotated.size,
        }
    return csv_filepath

class SatelliteProcessor:
    """Facade for satellite processing flow."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def process_satellite(self, tif_path, output_dir, scale=1.0, patch_size=500, overlap=100, **kwargs):
        return process_satellite_as_drone(
            tif_path=tif_path,
            output_dir=output_dir,
            model=self.model,
            device=self.device,
            scale=scale,
            patch_size=patch_size,
            overlap=overlap,
            **kwargs,
        )


class UAVProcessor:
    """Facade for UAV Ekeland feature extraction flow."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def extract_uav_features(self, uav_img_path, uav_csv_path, method='marching_squares', max_search_radius=100.0, output_dir=None):
        return complete_segmentation_demo_uav_with_rotation(
            model=self.model,
            device=self.device,
            uav_img_path=uav_img_path,
            uav_csv_path=uav_csv_path,
            method=method,
            max_search_radius=max_search_radius,
            output_dir=output_dir,
        )

if __name__ == "__main__":
    DATA_ROOT = "/home/nguyenduytan/UAV_nonGPS/UAV_nonGPS_dataset"
    FLIGHT_ID = "03"

    MODEL_PATH = "/home/nguyenduytan/UAV_nonGPS/best_model.pth"

    FLIGHT_CSV = os.path.join(DATA_ROOT, FLIGHT_ID, f"{FLIGHT_ID}.csv")
    UAV_IMG_PATH = os.path.join(DATA_ROOT, FLIGHT_ID, "drone", "03_0049.JPG")
    SAT_IMG_PATH = os.path.join(DATA_ROOT, FLIGHT_ID, f"satellite{FLIGHT_ID}.tif")

    OUT_DIR = os.path.join(DATA_ROOT, FLIGHT_ID, "output")
    os.makedirs(OUT_DIR, exist_ok=True)

    SAT_CSV_PATH = os.path.join(OUT_DIR, f"satellite{FLIGHT_ID}.csv") 
    print(FLIGHT_CSV, UAV_IMG_PATH, SAT_IMG_PATH, SAT_CSV_PATH, sep="\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path=MODEL_PATH,
        device=device,
        num_classes=2,
        pretrained=False,   
    ).to(device).eval()

    print("Device:", device)

    UAV_CSV_ROTATED = complete_segmentation_demo_uav_with_rotation(
        model=model,
        device=device,
        uav_img_path=UAV_IMG_PATH,
        uav_csv_path=FLIGHT_CSV,
        method="marching_squares",
        max_search_radius=100.0,
        output_dir=OUT_DIR,
        max_vis_ekeland_triangles=1
    )
    print("UAV CSV:", UAV_CSV_ROTATED)

# PYTHONPATH=. python ./satellite/preprocess.py