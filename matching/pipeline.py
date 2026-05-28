import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..dataset import process_uav
from ..satellite.preprocess import (
    compare_uav_satellite_segmentation,
    complete_segmentation_demo_uav_with_rotation,
    complete_segmentation_demo_uav_with_rotation_multiscale,
    generate_satellite_soft_mask,
    generate_satellite_soft_mask_crop,
)
from .features import latlon_to_pixel, load_satellite_bounds, process_matching


def _find_uav_pose_row(flight_csv, uav_img_path):
    """Find UAV metadata row by image filename."""
    df = pd.read_csv(flight_csv)
    uav_name = os.path.basename(uav_img_path).strip().lower()
    matches = df[df["filename"].astype(str).str.strip().str.lower() == uav_name]
    if matches.empty:
        raise ValueError(f"Khong tim thay '{uav_name}' trong flight CSV: {flight_csv}")
    return matches.iloc[0]


def _load_satellite_mask_from_path(mask_path):
    """Load satellite mask from .npy or image file."""
    ext = os.path.splitext(mask_path)[1].lower()
    if ext == ".npy":
        return np.load(mask_path)

    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Khong doc duoc satellite mask: {mask_path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def _skipped_mask_compare_metrics(reason):
    """Return a consistent payload when mask compare is skipped."""
    return {
        "status": "skipped",
        "reason": reason,
        "metric_mode": None,
        "threshold_source": None,
        "threshold": None,
    }


def _compute_centered_crop_bbox(center_x, center_y, crop_w, crop_h, image_w, image_h):
    """Build a clipped crop bbox centered around one pixel location."""
    crop_w = int(max(1, crop_w))
    crop_h = int(max(1, crop_h))

    x0 = int(round(center_x - crop_w / 2))
    y0 = int(round(center_y - crop_h / 2))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > image_w:
        shift = x1 - image_w
        x0 = max(0, x0 - shift)
        x1 = image_w
    if y1 > image_h:
        shift = y1 - image_h
        y0 = max(0, y0 - shift)
        y1 = image_h

    return [int(x0), int(y0), int(x1), int(y1)]


def _pixel_to_latlon(px, py, bounds, sat_width, sat_height):
    """Convert satellite pixel (x, y) back to geographic coordinates."""
    x_norm = float(px) / float(max(sat_width - 1, 1))
    y_norm = float(py) / float(max(sat_height - 1, 1))
    lat = float(bounds["LT_lat"] - y_norm * (bounds["LT_lat"] - bounds["RB_lat"]))
    lon = float(bounds["LT_lon"] + x_norm * (bounds["RB_lon"] - bounds["LT_lon"]))
    return lat, lon


class LocalizationPipeline:
    """OOP orchestrator for the end-to-end localization flow."""

    def __init__(self, model=None, device=None):
        self.model = model
        self.device = device
        self.last_mask_compare_metrics = None

    def _compare_masks_at_gt_position(
        self,
        *,
        gt_lat,
        gt_lon,
        uav_img_path,
        sat_img_path,
        output_dir,
        uav_seg_result,
        bounds=None,
        satellite_mask_path=None,
        compare_threshold=0.5,
        compare_metric_mode="binary",
        satellite_mask_scale=1.0,
        save_compare_figure=True,
        use_local_satellite_crop=True,
        satellite_crop_margin_scale=1.0,
        satellite_patch_size=500,
        satellite_overlap=100,
        satellite_dynamic_threshold=True,
        satellite_dynamic_quantile=0.6,
        satellite_min_seg_threshold=0.35,
        satellite_max_seg_threshold=0.85,
        satellite_use_uncertainty_filter=True,
        satellite_uncertainty_ratio=0.5,
        enable_local_offset_search=False,
        local_offset_max_px=120,
        local_offset_step_px=20,
        local_offset_metric="iou",
        strict=False,
    ):
        """Run crop-and-compare mask logic around UAV GT position."""
        if not isinstance(uav_seg_result, dict):
            reason = "Khong nhan duoc ket qua segmentation UAV hop le."
            self.last_mask_compare_metrics = _skipped_mask_compare_metrics(reason)
            if strict:
                raise ValueError(reason)
            print(f"  Bo qua compare: {reason}")
            return self.last_mask_compare_metrics

        compare_metric_mode = str(compare_metric_mode).strip().lower()
        if compare_metric_mode not in {"binary", "soft"}:
            raise ValueError("compare_metric_mode must be 'binary' or 'soft'")

        if bounds is None:
            bounds = load_satellite_bounds(os.path.basename(sat_img_path))

        if bounds is None:
            reason = f"Khong doc duoc satellite bounds cho '{os.path.basename(sat_img_path)}'."
            self.last_mask_compare_metrics = _skipped_mask_compare_metrics(reason)
            if strict:
                raise ValueError(reason)
            print(f"  Bo qua compare: {reason}")
            return self.last_mask_compare_metrics

        uav_mask_key_candidates = ["uav_soft_mask", "uav_mask"]
        uav_mask_key = None
        uav_compare_mask = None
        for key in uav_mask_key_candidates:
            value = uav_seg_result.get(key)
            if value is not None:
                uav_mask_key = key
                uav_compare_mask = value
                break
        if uav_compare_mask is None:
            reason = "UAV compare mask rong."
            self.last_mask_compare_metrics = _skipped_mask_compare_metrics(reason)
            if strict:
                raise ValueError(reason)
            print(f"  Bo qua compare: {reason}")
            return self.last_mask_compare_metrics

        uav_h, uav_w = np.asarray(uav_compare_mask).shape[:2]
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(sat_img_path) as sat_img:
            sat_img_w, sat_img_h = sat_img.size

        center_x_full, center_y_full = latlon_to_pixel(
            gt_lat,
            gt_lon,
            bounds,
            sat_width=sat_img_w,
            sat_height=sat_img_h,
        )

        compare_output_path = None
        if save_compare_figure:
            base_name = os.path.splitext(os.path.basename(uav_img_path))[0]
            compare_output_path = os.path.join(output_dir, f"mask_compare_{base_name}.png")

        crop_context_bbox = None
        local_center_x, local_center_y = center_x_full, center_y_full
        if satellite_mask_path:
            sat_soft_mask = _load_satellite_mask_from_path(satellite_mask_path)
        elif use_local_satellite_crop:
            crop_w = max(1, int(round(uav_w * float(satellite_crop_margin_scale))))
            crop_h = max(1, int(round(uav_h * float(satellite_crop_margin_scale))))
            crop_context_bbox = _compute_centered_crop_bbox(
                center_x=center_x_full,
                center_y=center_y_full,
                crop_w=crop_w,
                crop_h=crop_h,
                image_w=sat_img_w,
                image_h=sat_img_h,
            )
            sat_soft_mask = generate_satellite_soft_mask_crop(
                sat_img_path=sat_img_path,
                crop_bbox_xyxy=crop_context_bbox,
                model=self.model,
                device=self.device,
                scale=satellite_mask_scale,
                patch_size=satellite_patch_size,
                overlap=satellite_overlap,
                dynamic_threshold=satellite_dynamic_threshold,
                dynamic_quantile=satellite_dynamic_quantile,
                min_seg_threshold=satellite_min_seg_threshold,
                max_seg_threshold=satellite_max_seg_threshold,
                use_uncertainty_filter=satellite_use_uncertainty_filter,
                uncertainty_ratio=satellite_uncertainty_ratio,
            )
            local_center_x = center_x_full - crop_context_bbox[0]
            local_center_y = center_y_full - crop_context_bbox[1]
        else:
            sat_soft_mask = generate_satellite_soft_mask(
                sat_img_path=sat_img_path,
                model=self.model,
                device=self.device,
                scale=satellite_mask_scale,
                patch_size=satellite_patch_size,
                overlap=satellite_overlap,
                dynamic_threshold=satellite_dynamic_threshold,
                dynamic_quantile=satellite_dynamic_quantile,
                min_seg_threshold=satellite_min_seg_threshold,
                max_seg_threshold=satellite_max_seg_threshold,
                use_uncertainty_filter=satellite_use_uncertainty_filter,
                uncertainty_ratio=satellite_uncertainty_ratio,
            )

        metrics = compare_uav_satellite_segmentation(
            uav_mask=uav_compare_mask,
            satellite_mask=sat_soft_mask,
            center_x=local_center_x,
            center_y=local_center_y,
            threshold=compare_threshold,
            output_path=compare_output_path,
            metric_mode=compare_metric_mode,
        )
        if enable_local_offset_search:
            metric_name = str(local_offset_metric).lower().strip()
            if metric_name not in {"iou", "dice"}:
                metric_name = "iou"
            search_radius = int(max(0, local_offset_max_px))
            search_step = int(max(1, local_offset_step_px))
            best_metrics = metrics
            best_local_center = (int(local_center_x), int(local_center_y))
            base_score = float(metrics.get(metric_name, 0.0))

            for dy in range(-search_radius, search_radius + 1, search_step):
                for dx in range(-search_radius, search_radius + 1, search_step):
                    if dx == 0 and dy == 0:
                        continue
                    cand_center_x = int(local_center_x + dx)
                    cand_center_y = int(local_center_y + dy)
                    cand_metrics = compare_uav_satellite_segmentation(
                        uav_mask=uav_compare_mask,
                        satellite_mask=sat_soft_mask,
                        center_x=cand_center_x,
                        center_y=cand_center_y,
                        threshold=compare_threshold,
                        output_path=None,
                        metric_mode=compare_metric_mode,
                    )
                    cand_score = float(cand_metrics.get(metric_name, 0.0))
                    if cand_score > float(best_metrics.get(metric_name, 0.0)):
                        best_metrics = cand_metrics
                        best_local_center = (cand_center_x, cand_center_y)

            local_center_x, local_center_y = best_local_center
            if compare_output_path:
                best_metrics = compare_uav_satellite_segmentation(
                    uav_mask=uav_compare_mask,
                    satellite_mask=sat_soft_mask,
                    center_x=local_center_x,
                    center_y=local_center_y,
                    threshold=compare_threshold,
                    output_path=compare_output_path,
                    metric_mode=compare_metric_mode,
                )
            metrics = best_metrics
            improved_score = float(metrics.get(metric_name, 0.0))
            print(
                f"  Local offset search ({metric_name}): "
                f"{base_score:.4f} -> {improved_score:.4f} "
                f"with step={search_step}px, radius={search_radius}px"
            )
        if crop_context_bbox is not None:
            local_crop_bbox = metrics["crop_bbox_xyxy"]
            metrics["crop_bbox_xyxy_local"] = local_crop_bbox
            metrics["satellite_context_bbox_xyxy"] = crop_context_bbox
            metrics["crop_bbox_xyxy"] = [
                int(local_crop_bbox[0] + crop_context_bbox[0]),
                int(local_crop_bbox[1] + crop_context_bbox[1]),
                int(local_crop_bbox[2] + crop_context_bbox[0]),
                int(local_crop_bbox[3] + crop_context_bbox[1]),
            ]
            metrics["center_xy_local"] = [int(local_center_x), int(local_center_y)]
            metrics["center_xy"] = [int(center_x_full), int(center_y_full)]
            metrics["refined_center_xy"] = [
                int(local_center_x + crop_context_bbox[0]),
                int(local_center_y + crop_context_bbox[1]),
            ]
        else:
            metrics["refined_center_xy"] = [int(local_center_x), int(local_center_y)]

        refined_lat, refined_lon = _pixel_to_latlon(
            px=metrics["refined_center_xy"][0],
            py=metrics["refined_center_xy"][1],
            bounds=bounds,
            sat_width=sat_img_w,
            sat_height=sat_img_h,
        )
        metrics["refined_lat"] = float(refined_lat)
        metrics["refined_lon"] = float(refined_lon)
        metrics["uav_mask_source"] = uav_mask_key
        self.last_mask_compare_metrics = metrics
        threshold_value = metrics.get("threshold")
        threshold_str = f"{threshold_value:.3f}" if threshold_value is not None else "n/a"
        print(
            "  Similarity:"
            f" IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f},"
            f" Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
        )
        print(
            "  Debug:"
            f" metric_mode={metrics.get('metric_mode', compare_metric_mode)},"
            f" uav_mask_source={uav_mask_key},"
            f" center=({metrics['center_xy'][0]},{metrics['center_xy'][1]}),"
            f" bbox={metrics['crop_bbox_xyxy']},"
            f" threshold_source={metrics.get('threshold_source', 'fixed')},"
            f" sat_crop[min,max,mean,q95]="
            f"({metrics['sat_crop_min']:.4f}, {metrics['sat_crop_max']:.4f},"
            f" {metrics['sat_crop_mean']:.4f}, {metrics['sat_crop_q95']:.4f}),"
            f" threshold={threshold_str}"
        )
        if crop_context_bbox is not None:
            print(f"  Satellite context bbox: {crop_context_bbox}")
        if compare_output_path:
            print(f"  Da luu anh compare: {compare_output_path}")
        return metrics

    def run(
        self,
        flight_csv,
        uav_img_path,
        uav_csv_path_raw,
        sat_img_path,
        sat_csv_path,
        top_k=1000,
        output_dir="/home/nguyenduytan/UAV_nonGPS/UAV_nonGPS_dataset/03/output",
        enable_mask_crop_compare=False,
        satellite_mask_path=None,
        compare_threshold=0.5,
        compare_metric_mode="binary",
        satellite_mask_scale=1.0,
        save_compare_figure=True,
        use_local_satellite_crop=True,
        satellite_crop_margin_scale=1.0,
        satellite_patch_size=500,
        satellite_overlap=100,
        # Shared segmentation behavior (align UAV and Satellite extraction).
        segmentation_dynamic_threshold=True,
        segmentation_dynamic_quantile=0.6,
        segmentation_min_seg_threshold=0.35,
        segmentation_max_seg_threshold=0.85,
        segmentation_use_uncertainty_filter=True,
        segmentation_uncertainty_ratio=0.5,
        # --- UAV multi-scale parameters ---
        uav_scales=[0.75, 1.0, 1.25],
        uav_scale_weights=[0.2, 0.6, 0.2],
        uav_soft_mask_scale=1.0,
        uav_patch_size=500,
        uav_overlap=100,
        uav_enable_otsu_cleanup=True,
        uav_morph_open_kernel=3,
        uav_morph_close_kernel=5,
        uav_min_component_area=50,
        enable_local_offset_search=False,
        local_offset_max_px=120,
        local_offset_step_px=20,
        local_offset_metric="iou",
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Da tao thu muc dau ra: {output_dir}")

        print('Step 1: Xoay anh UAV ve North-up...')

        try:
            _, img_rotated, _meta = process_uav(
                img_path=uav_img_path,
                csv_path=uav_csv_path_raw,
            )

            rotated_path = os.path.join(output_dir, 'uav_rotated.jpg')
            img_rotated.save(rotated_path, quality=95)
            print(f"  Da luu anh xoay thanh cong: {rotated_path}")

        except Exception as e:
            print(f"  Loi trong qua trinh xu ly/luu anh: {e}")
            rotated_path = None

        print('\nStep 2: Trich xuat dac trung UAV...')
        pose_row = _find_uav_pose_row(flight_csv=flight_csv, uav_img_path=uav_img_path)
        gt_lat, gt_lon = float(pose_row['lat']), float(pose_row['lon'])

        bounds = load_satellite_bounds(os.path.basename(sat_img_path))

        if self.model is None or self.device is None:
            raise ValueError('LocalizationPipeline requires model and device for UAV segmentation/extraction.')

        # --- GỌI HÀM MULTI-SCALE MỚI VÀ TRUYỀN THAM SỐ ---
        uav_seg_result = complete_segmentation_demo_uav_with_rotation_multiscale(
            model=self.model,
            device=self.device,
            uav_img_path=uav_img_path,
            uav_csv_path=uav_csv_path_raw,
            return_details=enable_mask_crop_compare,
            scales=uav_scales,
            scale_weights=uav_scale_weights,
            patch_size=uav_patch_size,
            overlap=uav_overlap,
            dynamic_threshold=segmentation_dynamic_threshold,
            dynamic_quantile=segmentation_dynamic_quantile,
            min_seg_threshold=segmentation_min_seg_threshold,
            max_seg_threshold=segmentation_max_seg_threshold,
            use_uncertainty_filter=segmentation_use_uncertainty_filter,
            uncertainty_ratio=segmentation_uncertainty_ratio,
            enable_otsu_cleanup=uav_enable_otsu_cleanup,
            morph_open_kernel=uav_morph_open_kernel,
            morph_close_kernel=uav_morph_close_kernel,
            min_component_area=uav_min_component_area,
            soft_mask_scale_for_compare=uav_soft_mask_scale,
        )
        
        # Xử lý kết quả trả về (giữ nguyên logic cũ)
        uav_csv_rotated = uav_seg_result.get("csv_path") if enable_mask_crop_compare else uav_seg_result
        
        self.last_mask_compare_metrics = None
        matching_lat, matching_lon = gt_lat, gt_lon
        if enable_mask_crop_compare:
            print('\nStep 3: Crop satellite mask theo vi tri UAV va so sanh segmentation...')
            metrics = self._compare_masks_at_gt_position(
                gt_lat=gt_lat,
                gt_lon=gt_lon,
                uav_img_path=uav_img_path,
                sat_img_path=sat_img_path,
                output_dir=output_dir,
                uav_seg_result=uav_seg_result,
                bounds=bounds,
                satellite_mask_path=satellite_mask_path,
                compare_threshold=compare_threshold,
                compare_metric_mode=compare_metric_mode,
                satellite_mask_scale=satellite_mask_scale,
                save_compare_figure=save_compare_figure,
                use_local_satellite_crop=use_local_satellite_crop,
                satellite_crop_margin_scale=satellite_crop_margin_scale,
                satellite_patch_size=satellite_patch_size,
                satellite_overlap=satellite_overlap,
                satellite_dynamic_threshold=segmentation_dynamic_threshold,
                satellite_dynamic_quantile=segmentation_dynamic_quantile,
                satellite_min_seg_threshold=segmentation_min_seg_threshold,
                satellite_max_seg_threshold=segmentation_max_seg_threshold,
                satellite_use_uncertainty_filter=segmentation_use_uncertainty_filter,
                satellite_uncertainty_ratio=segmentation_uncertainty_ratio,
                enable_local_offset_search=enable_local_offset_search,
                local_offset_max_px=local_offset_max_px,
                local_offset_step_px=local_offset_step_px,
                local_offset_metric=local_offset_metric,
                strict=False,
            )
            if metrics and metrics.get("status") != "skipped":
                matching_lat = float(metrics.get("refined_lat", matching_lat))
                matching_lon = float(metrics.get("refined_lon", matching_lon))

        results = []
        if not uav_csv_rotated:
            print("  [Cảnh báo] Không trích xuất được UAV CSV (không có polygon hợp lệ). Bỏ qua Step 2 (Matching).")
        else:
            # Matching dùng tâm đã refine (nếu có) để đánh giá/visualize ổn định hơn.
            print('\nStep 2: Tien hanh so khop dac trung...')
            results = process_matching(
                uav_csv_path=uav_csv_rotated,
                sat_csv_path=sat_csv_path,
                uav_img_path=uav_img_path,
                sat_img_path=sat_img_path,
                top_k=top_k,
                gt_lat=matching_lat,
                gt_lon=matching_lon,
                sat_bounds=bounds,
                uav_rotated_img_path=rotated_path,
            )
        return results

    def run_mask_crop_compare_only(
        self,
        flight_csv,
        uav_img_path,
        sat_img_path,
        uav_csv_path_raw=None,
        output_dir="/home/nguyenduytan/UAV_nonGPS/UAV_nonGPS_dataset/03/output",
        satellite_mask_path=None,
        compare_threshold=0.5,
        compare_metric_mode="binary",
        satellite_mask_scale=1.0,
        save_compare_figure=True,
        use_local_satellite_crop=True,
        satellite_crop_margin_scale=1.0,
        satellite_patch_size=500,
        satellite_overlap=100,
        segmentation_dynamic_threshold=True,
        segmentation_dynamic_quantile=0.6,
        segmentation_min_seg_threshold=0.35,
        segmentation_max_seg_threshold=0.85,
        segmentation_use_uncertainty_filter=True,
        segmentation_uncertainty_ratio=0.5,
        uav_scales=[0.75],
        uav_scale_weights=[1.0],
        uav_soft_mask_scale=1.0,
        uav_patch_size=500,
        uav_overlap=100,
        uav_enable_otsu_cleanup=True,
        uav_morph_open_kernel=3,
        uav_morph_close_kernel=5,
        uav_min_component_area=50,
        enable_local_offset_search=False,
        local_offset_max_px=120,
        local_offset_step_px=20,
        local_offset_metric="iou",
    ):
        """Run only the Step-3 mask crop/compare flow without triangle matching."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Da tao thu muc dau ra: {output_dir}")

        if self.model is None or self.device is None:
            raise ValueError('LocalizationPipeline requires model and device for UAV segmentation/extraction.')

        if uav_csv_path_raw is None:
            uav_csv_path_raw = flight_csv

        pose_row = _find_uav_pose_row(flight_csv=flight_csv, uav_img_path=uav_img_path)
        gt_lat, gt_lon = float(pose_row['lat']), float(pose_row['lon'])

        print('Step 3 only: Crop satellite mask theo vi tri UAV va so sanh segmentation...')
        uav_seg_result = complete_segmentation_demo_uav_with_rotation_multiscale(
            model=self.model,
            device=self.device,
            uav_img_path=uav_img_path,
            uav_csv_path=uav_csv_path_raw,
            return_details=True,
            scales=uav_scales,
            scale_weights=uav_scale_weights,
            patch_size=uav_patch_size,
            overlap=uav_overlap,
            dynamic_threshold=segmentation_dynamic_threshold,
            dynamic_quantile=segmentation_dynamic_quantile,
            min_seg_threshold=segmentation_min_seg_threshold,
            max_seg_threshold=segmentation_max_seg_threshold,
            use_uncertainty_filter=segmentation_use_uncertainty_filter,
            uncertainty_ratio=segmentation_uncertainty_ratio,
            enable_otsu_cleanup=uav_enable_otsu_cleanup,
            morph_open_kernel=uav_morph_open_kernel,
            morph_close_kernel=uav_morph_close_kernel,
            min_component_area=uav_min_component_area,
            soft_mask_scale_for_compare=uav_soft_mask_scale,
        )

        return self._compare_masks_at_gt_position(
            gt_lat=gt_lat,
            gt_lon=gt_lon,
            uav_img_path=uav_img_path,
            sat_img_path=sat_img_path,
            output_dir=output_dir,
            uav_seg_result=uav_seg_result,
            bounds=None,
            satellite_mask_path=satellite_mask_path,
            compare_threshold=compare_threshold,
            compare_metric_mode=compare_metric_mode,
            satellite_mask_scale=satellite_mask_scale,
            save_compare_figure=save_compare_figure,
            use_local_satellite_crop=use_local_satellite_crop,
            satellite_crop_margin_scale=satellite_crop_margin_scale,
            satellite_patch_size=satellite_patch_size,
            satellite_overlap=satellite_overlap,
            satellite_dynamic_threshold=segmentation_dynamic_threshold,
            satellite_dynamic_quantile=segmentation_dynamic_quantile,
            satellite_min_seg_threshold=segmentation_min_seg_threshold,
            satellite_max_seg_threshold=segmentation_max_seg_threshold,
            satellite_use_uncertainty_filter=segmentation_use_uncertainty_filter,
            satellite_uncertainty_ratio=segmentation_uncertainty_ratio,
            enable_local_offset_search=enable_local_offset_search,
            local_offset_max_px=local_offset_max_px,
            local_offset_step_px=local_offset_step_px,
            local_offset_metric=local_offset_metric,
            strict=True,
        )


def run_full_pipeline(
    flight_csv,
    uav_img_path,
    uav_csv_path_raw,
    sat_img_path,
    sat_csv_path,
    top_k=1000,
    output_dir="/home/nguyenduytan/UAV_nonGPS/UAV_nonGPS_dataset/03/output",
    **kwargs,
):
    """Backward-compatible functional API (legacy notebooks)."""
    pipeline = LocalizationPipeline(model=globals().get('model'), device=globals().get('device'))
    return pipeline.run(
        flight_csv=flight_csv,
        uav_img_path=uav_img_path,
        uav_csv_path_raw=uav_csv_path_raw,
        sat_img_path=sat_img_path,
        sat_csv_path=sat_csv_path,
        top_k=top_k,
        output_dir=output_dir,
        **kwargs,
    )
