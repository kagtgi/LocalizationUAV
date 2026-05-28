import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import os
import heapq
import re
from pathlib import Path

from ..dataset import process_uav
from ..satellite.preprocess import complete_segmentation_demo_uav_with_rotation

def dist(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_internal_angles(v0, v1, v2):
    """
    Tính 3 góc trong của tam giác từ tọa độ 3 đỉnh.
    Returns: list 3 góc theo độ, sắp xếp tăng dần.
    """
    a = dist(v1, v2)
    b = dist(v0, v2)
    c = dist(v0, v1)

    if a * b == 0 or b * c == 0 or c * a == 0:
        return [0, 0, 0]

    try:
        angle_A = math.degrees(math.acos(max(-1.0, min(1.0, (b**2 + c**2 - a**2) / (2*b*c)))))
        angle_B = math.degrees(math.acos(max(-1.0, min(1.0, (a**2 + c**2 - b**2) / (2*a*c)))))
        angle_C = 180.0 - angle_A - angle_B
    except ValueError:
        return [0, 0, 0]

    return sorted([angle_A, angle_B, angle_C])  # Sắp xếp tăng dần


def extract_features_from_row(row,
                              include_shape_features=False,
                              include_side_ratios=True,
                              include_radius_ratio=True,
                              include_elongation=True,
                              eps=1e-8):
    """
    Trích xuất vector đặc trưng 5 chiều từ một dòng CSV:
      [geo_angle_1, geo_angle_2, ekeland_1, ekeland_2, ekeland_3]
    Tất cả đã sắp xếp tăng dần → bất biến với hoán vị đỉnh.
    """
    v0 = (row['vertex_0_x'], row['vertex_0_y'])
    v1 = (row['vertex_1_x'], row['vertex_1_y'])
    v2 = (row['vertex_2_x'], row['vertex_2_y'])

    # 2 góc trong (lấy 2 góc nhỏ nhất, góc thứ 3 = 180 - a - b)
    geo_angles = calculate_internal_angles(v0, v1, v2)
    geo_feats = geo_angles[:2]  # [góc nhỏ nhất, góc thứ 2]

    # 3 góc Ekeland (sắp xếp tăng dần)
    eke_angles = sorted([row['ekeland_v0'], row['ekeland_v1'], row['ekeland_v2']])

    features = geo_feats + eke_angles  # base 5-D: 2 interior + 3 ekeland

    if not include_shape_features:
        return features

    # Side lengths
    a = dist(v1, v2)
    b = dist(v0, v2)
    c = dist(v0, v1)
    s0, s1, s2 = sorted([a, b, c])

    if include_side_ratios:
        features.extend([
            s0 / (s1 + eps),
            s1 / (s2 + eps),
            s0 / (s2 + eps),
        ])

    if include_radius_ratio:
        # Heron area (robust)
        semi = 0.5 * (a + b + c)
        area_sq = max(semi * (semi - a) * (semi - b) * (semi - c), 0.0)
        area = math.sqrt(area_sq)
        # r = A/s, R = abc/(4A) -> r/R = 4A^2/(sabc)
        denom = max(semi * a * b * c, eps)
        features.append((4.0 * area * area) / denom)

    if include_elongation:
        features.append(s2 / (s0 + eps))

    return features


def l1_distance(feat_a, feat_b):
    """Tính khoảng cách L1 giữa 2 vector đặc trưng"""
    return sum(abs(a - b) for a, b in zip(feat_a, feat_b))

def l2_distance(feat_a, feat_b):
    """Tính khoảng cách L2 giữa 2 vector đặc trưng"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(feat_a, feat_b)))

def complex_angle_distance(feat_a, feat_b):
    """
    Tính khoảng cách giữa 2 vector góc dựa trên mặt phẳng phức (Chordal Distance).
    feat_a, feat_b: list các góc (độ).
    """
    dist_sum = 0
    for a, b in zip(feat_a, feat_b):
        # Chuyển sang radian
        rad_a, rad_b = np.radians(a), np.radians(b)
        
        # Biểu diễn số phức: z = cos(alpha) + i*sin(alpha)
        # Khoảng cách L2 giữa 2 số phức trên vòng tròn đơn vị
        d = np.sqrt(2 * (1 - np.cos(rad_a - rad_b)))
        dist_sum += d
        
    return dist_sum

def angular_distance(a, b):
    """
    Tính khoảng cách ngắn nhất giữa hai góc (độ) trên vòng tròn 360.
    Kết quả luôn nằm trong khoảng [0, 180].
    """
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff

def geodesic_feature_distance(feat_a, feat_b):
    """
    Tính tổng khoảng cách Geodesic cho vector đặc trưng (5 chiều).
    Thay thế cho hàm l1_distance.
    """
    return sum(angular_distance(a, b) for a, b in zip(feat_a, feat_b))

def circular_mean_distance(feat_a, feat_b):
    """
    Circular mean distance cho vector góc.
    Mỗi chiều dùng: d = 1 - cos(delta), với delta là sai khác góc ngắn nhất.
    Trả về trung bình trên toàn bộ chiều (0 là giống nhau hoàn toàn).
    """
    circ_dists = []
    for a, b in zip(feat_a, feat_b):
        delta_deg = angular_distance(a, b)
        delta_rad = np.radians(delta_deg)
        circ_dists.append(1.0 - np.cos(delta_rad))
    return float(np.mean(circ_dists))

def von_mises_feature_score(feat_a, feat_b, kappa=4.0):
    """
    Von Mises-based score (negative log-likelihood, bỏ hằng số chuẩn hóa theo từng chiều).
    Nhỏ hơn => giống nhau hơn.

    kappa càng lớn thì phạt sai khác góc càng mạnh.
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0 for von Mises score")

    nll = 0.0
    for a, b in zip(feat_a, feat_b):
        delta_deg = angular_distance(a, b)
        delta_rad = np.radians(delta_deg)
        # Full NLL của von Mises theo từng chiều:
        # -log p(delta) = -kappa*cos(delta) + log(2*pi*I0(kappa))
        nll += (-kappa * np.cos(delta_rad)) + np.log(2.0 * np.pi * np.i0(kappa))
    return float(nll)

def compute_feature_distance(feat_a, feat_b, metric='complex', vm_kappa=4.0,
                             angle_feature_dims=None, linear_weight=1.0):
    """
    Chọn metric khoảng cách khi matching.
    Supported: l1, l2, complex, geodesic, circular_mean, von_mises
    """
    metric = metric.lower()
    if angle_feature_dims is None:
        angle_feature_dims = min(len(feat_a), len(feat_b))

    angle_a = feat_a[:angle_feature_dims]
    angle_b = feat_b[:angle_feature_dims]
    linear_a = feat_a[angle_feature_dims:]
    linear_b = feat_b[angle_feature_dims:]
    if metric == 'l1':
        return l1_distance(feat_a, feat_b)
    if metric == 'l2':
        return l2_distance(feat_a, feat_b)
    if metric == 'complex':
        score = complex_angle_distance(angle_a, angle_b)
        if linear_a and linear_b:
            score += linear_weight * l1_distance(linear_a, linear_b)
        return score
    if metric == 'geodesic':
        score = geodesic_feature_distance(angle_a, angle_b)
        if linear_a and linear_b:
            score += linear_weight * l1_distance(linear_a, linear_b)
        return score
    if metric == 'circular_mean':
        score = circular_mean_distance(angle_a, angle_b)
        if linear_a and linear_b:
            score += linear_weight * l1_distance(linear_a, linear_b)
        return score
    if metric == 'von_mises':
        score = von_mises_feature_score(angle_a, angle_b, kappa=vm_kappa)
        if linear_a and linear_b:
            score += linear_weight * l1_distance(linear_a, linear_b)
        return score
    raise ValueError(f"Unsupported metric: {metric}")

def process_matching(uav_csv_path, 
                     sat_csv_path, 
                     uav_img_path, 
                     sat_img_path,
                     top_k=1000, 
                     uav_rotated_img_path=None,
                     gt_lat=None, 
                     gt_lon=None, 
                     gt_alt=None, 
                     gt_yaw=None,
                     sat_bounds=None, 
                     metric='l1', 
                     vm_kappa=4.0,
                     include_shape_features=False, 
                     include_side_ratios=True,
                     include_radius_ratio=True, 
                     include_elongation=True,
                     linear_weight=1.0): 
    """
    So khớp đặc trưng Ekeland + vẽ Ground Truth lên Satellite
    """
    print("=" * 60)
    print("EKELAND MATCHING PIPELINE + GROUND TRUTH")
    print("=" * 60)

    # 1. Đọc dữ liệu
    df_uav = pd.read_csv(uav_csv_path)
    df_sat = pd.read_csv(sat_csv_path)

    print(f"UAV triangles: {len(df_uav)} | SAT triangles: {len(df_sat)}")

    # 2. Tính features
    uav_features = [
        extract_features_from_row(
            row,
            include_shape_features=include_shape_features,
            include_side_ratios=include_side_ratios,
            include_radius_ratio=include_radius_ratio,
            include_elongation=include_elongation,
        )
        for _, row in df_uav.iterrows()
    ]
    sat_features = [
        extract_features_from_row(
            row,
            include_shape_features=include_shape_features,
            include_side_ratios=include_side_ratios,
            include_radius_ratio=include_radius_ratio,
            include_elongation=include_elongation,
        )
        for _, row in df_sat.iterrows()
    ]

    angle_feature_dims = 5  # 2 góc trong + 3 góc Ekeland

    # 3. Brute-force matching
    if top_k <= 0:
        return []

    # Keep only best top_k candidates in a max-heap encoded by -score.
    top_heap = []
    for idx_u, feat_u in enumerate(uav_features):
        for idx_s, feat_s in enumerate(sat_features):
            score = compute_feature_distance(
                feat_u, feat_s,
                metric=metric,
                vm_kappa=vm_kappa,
                angle_feature_dims=angle_feature_dims,
                linear_weight=linear_weight
            ) # Khoảng cách càng nhỏ -> càng giống
            candidate = (-score, idx_u, idx_s)
            if len(top_heap) < top_k:
                heapq.heappush(top_heap, candidate)
            elif score < -top_heap[0][0]:
                heapq.heapreplace(top_heap, candidate)

    top_candidates = sorted(top_heap, key=lambda x: -x[0])  # ascending score
    top_matches = [
        {
            'score': -neg_score,
            'uav_idx': idx_u,
            'sat_idx': idx_s,
            'uav_row': df_uav.iloc[idx_u],
            'sat_row': df_sat.iloc[idx_s],
        }
        for neg_score, idx_u, idx_s in top_candidates
    ]

    # 4. In kết quả
    print(f"\nTop {top_k} matches:")
    for i, m in enumerate(top_matches):
        print(f"  Top {i+1}: Score={m['score']:.3f} | UAV#{m['uav_idx']} → SAT#{m['sat_idx']}")
    if top_matches:
        unique_uav = len({m['uav_idx'] for m in top_matches})
        unique_sat = len({m['sat_idx'] for m in top_matches})
        total_pairs = len(df_uav) * len(df_sat)
        print(
            f"  Unique triangles in Top {top_k}: UAV={unique_uav}/{len(df_uav)} | "
            f"SAT={unique_sat}/{len(top_matches)}"
        )
        print(
            f"  Note: Top {top_k} được chọn trên toàn bộ {total_pairs:,} cặp UAV×SAT, "
            "nên một tam giác UAV có thể xuất hiện nhiều lần với các tam giác SAT khác nhau."
        )

    # 5. Visualize (truyền GT xuống)
    display_uav_path = uav_rotated_img_path if (uav_rotated_img_path and os.path.exists(uav_rotated_img_path)) else uav_img_path
    
    visualize_matches(
        display_uav_path, sat_img_path, top_matches,
        uav_is_rotated=(display_uav_path == uav_rotated_img_path),
        gt_lat=gt_lat,
        gt_lon=gt_lon,
        gt_alt=gt_alt,   # Truyền Alt
        gt_yaw=gt_yaw,   # Truyền Yaw
        sat_bounds=sat_bounds
    )

    return top_matches


# ==============================================================================
# VISUALIZATION: Vẽ kết quả so khớp lên cả 2 ảnh
# ==============================================================================
def _default_satellite_bounds_csv_candidates():
    """Return likely bounds CSV locations in priority order."""
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "UAV_nonGPS_dataset"
    return [
        dataset_root / "satellite_coordinates_range.csv",
        dataset_root / "satellite_ coordinates_range.csv",
    ]


def _normalize_satellite_name(name):
    """Normalize satellite map/image names for robust matching."""
    stem = os.path.splitext(os.path.basename(str(name)))[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _extract_satellite_numeric_id(name):
    """Extract trailing numeric id if present, e.g. satellite03 -> 3."""
    normalized = _normalize_satellite_name(name)
    matches = re.findall(r"\d+", normalized)
    if not matches:
        return None
    numeric = matches[-1].lstrip("0")
    return numeric or "0"


def _resolve_satellite_bounds_csv_path(csv_path=None):
    """Resolve bounds CSV path from explicit path or repo-local defaults."""
    candidates = [Path(csv_path)] if csv_path else _default_satellite_bounds_csv_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _match_satellite_bounds_row(df, satellite_filename):
    """Match one CSV row to a satellite image name despite naming differences."""
    map_col = df.columns[0]
    map_names = df[map_col].astype(str).fillna("")
    map_norm = map_names.map(_normalize_satellite_name)
    map_norm_no_prefix = map_norm.str.replace(r"^satellite", "", regex=True)

    target_norm = _normalize_satellite_name(satellite_filename)
    target_norm_no_prefix = re.sub(r"^satellite", "", target_norm)
    target_numeric = _extract_satellite_numeric_id(satellite_filename)

    exact_mask = (
        (map_norm == target_norm)
        | (map_norm == target_norm_no_prefix)
        | (map_norm_no_prefix == target_norm)
        | (map_norm_no_prefix == target_norm_no_prefix)
    )
    if exact_mask.any():
        return df.loc[exact_mask].iloc[0], map_col

    if target_numeric is not None:
        numeric_ids = map_names.map(_extract_satellite_numeric_id)
        numeric_mask = numeric_ids == target_numeric
        if numeric_mask.any():
            return df.loc[numeric_mask].iloc[0], map_col

    contains_mask = (
        map_norm.str.contains(target_norm, na=False)
        | map_norm.str.contains(target_norm_no_prefix, na=False)
        | map_norm_no_prefix.str.contains(target_norm_no_prefix, na=False)
    )
    if contains_mask.any():
        return df.loc[contains_mask].iloc[0], map_col

    return None, map_col


def load_satellite_bounds(satellite_filename, csv_path=None):
    resolved_csv_path = _resolve_satellite_bounds_csv_path(csv_path)
    if resolved_csv_path is None:
        print("=== DEBUG SATELLITE BOUNDS ===")
        print(" Khong tim thay file satellite bounds CSV.")
        print("============================")
        return None

    df = pd.read_csv(resolved_csv_path)

    print("=== DEBUG SATELLITE BOUNDS ===")
    print(f"CSV path: {resolved_csv_path}")
    print("Columns:", df.columns.tolist())
    print("First 3 rows:\n", df.head(3))

    row, map_col = _match_satellite_bounds_row(df, satellite_filename)
    if row is None:
        print(f" Khong tim thay satellite bounds cho '{satellite_filename}'")
        print("============================")
        return None

    print(f" Tìm thấy dòng: {row[map_col]}")

    bounds = {
        'LT_lat': float(row['LT_lat_map']),
        'LT_lon': float(row['LT_lon_map']),
        'RB_lat': float(row['RB_lat_map']),
        'RB_lon': float(row['RB_lon_map'])
    }
    print(f"Bounds: LT({bounds['LT_lat']}, {bounds['LT_lon']}) → RB({bounds['RB_lat']}, {bounds['RB_lon']})")
    print("============================")
    return bounds


def latlon_to_pixel(lat, lon, bounds, sat_width, sat_height, clip=True):
    """Convert lat/lon to pixel on satellite image using rectangular bounds."""
    lt_lat, lt_lon = bounds['LT_lat'], bounds['LT_lon']
    rb_lat, rb_lon = bounds['RB_lat'], bounds['RB_lon']

    lon_span = (rb_lon - lt_lon)
    lat_span = (lt_lat - rb_lat)
    if lon_span == 0 or lat_span == 0:
        raise ValueError("Invalid satellite bounds: zero span in lat/lon")

    x = int(round((lon - lt_lon) / lon_span * max(sat_width - 1, 1)))
    y = int(round((lt_lat - lat) / lat_span * max(sat_height - 1, 1)))  # y increases downward

    if clip:
        x = max(0, min(x, sat_width - 1))
        y = max(0, min(y, sat_height - 1))
    return x, y


def estimate_satellite_resolution_meters(bounds, sat_width, sat_height):
    """Approximate ground resolution from rectangular lat/lon bounds."""
    lt_lat, lt_lon = float(bounds['LT_lat']), float(bounds['LT_lon'])
    rb_lat, rb_lon = float(bounds['RB_lat']), float(bounds['RB_lon'])

    earth_radius_m = 6378137.0
    meters_per_deg_lat = math.pi * earth_radius_m / 180.0
    center_lat = 0.5 * (lt_lat + rb_lat)
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))

    width_m = abs(rb_lon - lt_lon) * meters_per_deg_lon
    height_m = abs(lt_lat - rb_lat) * meters_per_deg_lat
    denom_x = max(int(sat_width) - 1, 1)
    denom_y = max(int(sat_height) - 1, 1)

    return {
        'width_m': width_m,
        'height_m': height_m,
        'center_lat': center_lat,
        'm_per_px_x': width_m / denom_x,
        'm_per_px_y': height_m / denom_y,
        'm_per_px_mean': 0.5 * ((width_m / denom_x) + (height_m / denom_y)),
    }


def pixel_offset_to_meters(dx_px, dy_px, bounds, sat_width, sat_height):
    """Convert one pixel offset on the satellite image to approximate ground meters."""
    resolution = estimate_satellite_resolution_meters(bounds, sat_width, sat_height)
    dx_m = float(dx_px) * resolution['m_per_px_x']
    dy_m = float(dy_px) * resolution['m_per_px_y']
    return {
        **resolution,
        'dx_m': dx_m,
        'dy_m': dy_m,
        'distance_m': math.hypot(dx_m, dy_m),
    }


def log_bounds_pixel_sanity(bounds, sat_width, sat_height):
    """Print a quick sanity check for 4 bound corners mapped to pixels."""
    lt_lat, lt_lon = bounds['LT_lat'], bounds['LT_lon']
    rb_lat, rb_lon = bounds['RB_lat'], bounds['RB_lon']

    corners_geo = [
        ("LT", lt_lat, lt_lon),
        ("RT", lt_lat, rb_lon),
        ("RB", rb_lat, rb_lon),
        ("LB", rb_lat, lt_lon),
    ]

    expected = {
        "LT": (0, 0),
        "RT": (sat_width - 1, 0),
        "RB": (sat_width - 1, sat_height - 1),
        "LB": (0, sat_height - 1),
    }

    print("=== SANITY CHECK: Bounds -> Pixel ===")
    print("Assumption: north-up image + rectangular linear lat/lon mapping.")
    print(f"Expected near corners: LT(0,0), RT({sat_width-1},0), RB({sat_width-1},{sat_height-1}), LB(0,{sat_height-1})")
    max_corner_err = 0.0
    for name, lat, lon in corners_geo:
        px, py = latlon_to_pixel(lat, lon, bounds, sat_width, sat_height, clip=False)
        in_img = (0 <= px < sat_width) and (0 <= py < sat_height)
        ex, ey = expected[name]
        err = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
        max_corner_err = max(max_corner_err, err)
        print(f"  {name}: lat/lon=({lat:.6f},{lon:.6f}) -> pixel=({px},{py}) | in_image={in_img} | corner_err={err:.2f}")
    if max_corner_err > 3:
        print(f"WARNING: large corner error ({max_corner_err:.2f}px). Bounds or mapping assumption may be inconsistent.")
    print("=====================================")

def draw_match_triangle(img, row, color, rank, score=None, thickness_tri=3, marker_size=30):
    """
    Vẽ tam giác khớp lên ảnh: outline tam giác + marker tại trọng tâm + đỉnh + score.
    Trả về ảnh đã vẽ (in-place).
    """
    pts = np.array([
        [int(row['vertex_0_x']), int(row['vertex_0_y'])],
        [int(row['vertex_1_x']), int(row['vertex_1_y'])],
        [int(row['vertex_2_x']), int(row['vertex_2_y'])]
    ], np.int32).reshape((-1, 1, 2))

    # Vẽ outline tam giác
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness_tri)

    # Đánh dấu từng ĐỈNH của tam giác (star marker)
    for vi in ['vertex_0', 'vertex_1', 'vertex_2']:
        vx = int(row[f'{vi}_x'])
        vy = int(row[f'{vi}_y'])
        # Vẽ dấu "X" tại mỗi đỉnh để dễ nhận diện
        half = marker_size // 3
        cv2.line(img, (vx - half, vy - half), (vx + half, vy + half), color, 2)
        cv2.line(img, (vx + half, vy - half), (vx - half, vy + half), color, 2)
        # Vòng tròn nhỏ xung quanh đỉnh
        cv2.circle(img, (vx, vy), half + 2, color, 2)

    # Vẽ số thứ tự tại trọng tâm
    cx = int((row['vertex_0_x'] + row['vertex_1_x'] + row['vertex_2_x']) / 3)
    cy = int((row['vertex_0_y'] + row['vertex_1_y'] + row['vertex_2_y']) / 3)

    # Marker trọng tâm (màu đầy)
    cv2.circle(img, (cx, cy), marker_size, color, -1)
    cv2.circle(img, (cx, cy), marker_size + 2, (255, 255, 255), 2)  # Viền trắng
    # Số thứ tự
    text = str(rank)
    text_scale = 0.9 if marker_size >= 25 else 0.6
    text_thick = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thick)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thick)

    # Vẽ score dưới marker (nếu có)
    if score is not None:
        score_text = f"{score:.2f}"
        score_scale = 0.7
        score_thick = 1
        (sw, sh), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, score_scale, score_thick)
        score_y = cy + marker_size + 25
        cv2.putText(img, score_text, (cx - sw // 2, score_y),
                    cv2.FONT_HERSHEY_SIMPLEX, score_scale, color, score_thick)


def get_triangle_centroid(row):
    """Return triangle centroid in pixel coordinates."""
    cx = int(round((row['vertex_0_x'] + row['vertex_1_x'] + row['vertex_2_x']) / 3.0))
    cy = int(round((row['vertex_0_y'] + row['vertex_1_y'] + row['vertex_2_y']) / 3.0))
    return cx, cy


def save_match_centroid_preview(
    img_sat_rgb,
    matches,
    output_path,
    colors,
    highlight_rank=None,
    preview_max_dim=3200,
):
    """Save a compact preview with GT marker and small matched centroids."""
    if img_sat_rgb is None or len(matches) == 0:
        return None

    sat_h, sat_w = img_sat_rgb.shape[:2]
    scale = min(1.0, float(preview_max_dim) / float(max(sat_w, sat_h, 1)))
    preview_w = max(1, int(round(sat_w * scale)))
    preview_h = max(1, int(round(sat_h * scale)))
    if scale < 1.0:
        preview = cv2.resize(img_sat_rgb, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    else:
        preview = img_sat_rgb.copy()

    def _scale_point(point):
        return (
            int(round(point[0] * scale)),
            int(round(point[1] * scale)),
        )

    point_radius = max(3, min(8, min(preview_w, preview_h) // 220))
    font_scale = max(0.45, min(0.8, preview_w / 3000.0))
    text_thickness = max(1, int(round(font_scale * 2)))

    for i, match in enumerate(matches):
        centroid = get_triangle_centroid(match['sat_row'])
        px, py = _scale_point(centroid)
        color = colors[i % len(colors)]
        cv2.circle(preview, (px, py), point_radius + 1, (255, 255, 255), -1)
        cv2.circle(preview, (px, py), point_radius, color, -1)
        if i < 10 or (highlight_rank is not None and i + 1 == highlight_rank):
            cv2.putText(
                preview,
                str(i + 1),
                (px + point_radius + 3, py - point_radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                text_thickness,
            )

    legend_lines = [
        f"GT vs {len(matches)} matched centroids",
        "GT: red star | matches: small dots | yellow ring: nearest GT",
    ]
    for idx, line in enumerate(legend_lines):
        cv2.putText(
            preview,
            line,
            (20, 40 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
        )

    cv2.imwrite(output_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return output_path


# ==================== HÀM VISUALIZE ĐÃ SỬA + DEBUG ====================
def get_footprint_pixels(lat, lon, alt, yaw, hfov, bounds, sat_w, sat_h, aspect_ratio=4/3):
    """
    Tính toán 4 góc của footprint ảnh UAV trên tọa độ pixel của ảnh vệ tinh.
    """
    # 1. Tính kích thước vùng phủ tính bằng mét
    # W_m là chiều rộng ảnh, H_m là chiều cao ảnh trên mặt đất
    w_m = 2 * alt * math.tan(math.radians(hfov / 2))
    h_m = w_m / aspect_ratio
    
    # 2. Tọa độ 4 góc trong hệ quy chiếu cục bộ (m) so với tâm
    # Giả sử ảnh chưa xoay (North-up)
    corners_m = [
        (w_m/2, h_m/2),   # Top-Right
        (w_m/2, -h_m/2),  # Bottom-Right
        (-w_m/2, -h_m/2), # Bottom-Left
        (-w_m/2, h_m/2)   # Top-Left
    ]
    
    # 3. Xoay các góc theo Yaw của UAV
    # Yaw thường tính theo độ, cùng chiều kim đồng hồ từ hướng Bắc
    rad_yaw = math.radians(yaw)
    footprint_pts = []
    
    R_earth = 6378137.0 # Bán kính Trái Đất (mét)

    for dx, dy in corners_m:
        # Ma trận xoay
        rx = dx * math.cos(rad_yaw) + dy * math.sin(rad_yaw)
        ry = -dx * math.sin(rad_yaw) + dy * math.cos(rad_yaw)
        
        # 4. Chuyển mét sang độ lệch Lat/Lon (xấp xỉ)
        d_lat = (ry / R_earth) * (180.0 / math.pi)
        d_lon = (rx / (R_earth * math.cos(math.radians(lat)))) * (180.0 / math.pi)
        
        # 5. Chuyển sang tọa độ pixel trên ảnh vệ tinh
        px, py = latlon_to_pixel(lat + d_lat, lon + d_lon, bounds, sat_w, sat_h)
        footprint_pts.append([px, py])
        
    return np.array(footprint_pts, np.int32)


def visualize_matches(uav_img_path, sat_img_path, matches, 
                      uav_is_rotated=False,
                      gt_lat=None, gt_lon=None, gt_alt=None, gt_yaw=None, 
                      sat_bounds=None, hfov=94):
    
    img_uav = cv2.imread(uav_img_path)
    img_sat = cv2.imread(sat_img_path)

    if img_uav is None:
        print(f"Loi: Khong doc duoc anh UAV: {uav_img_path}")
        return
    if img_sat is None:
        print(f"Loi: Khong doc duoc anh Satellite: {sat_img_path}")
        return

    # Convert once to RGB for drawing with matplotlib.
    img_uav = cv2.cvtColor(img_uav, cv2.COLOR_BGR2RGB)
    img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)

    h, w = img_sat.shape[:2]
    print(f"Anh satellite size: {w}x{h}")

    gt_pixel = None
    best_i = -1
    min_dist = None

    # ==================== VE GROUND TRUTH & FOOTPRINT ====================
    if gt_lat is not None and gt_lon is not None and sat_bounds is not None:
        try:
            log_bounds_pixel_sanity(sat_bounds, w, h)

            # GT marker size: avoid oversized marker on very large image.
            dynamic_size = 300 # max(40, min(160, int(min(w, h) * 0.015)))
            dynamic_thickness = max(2, min(12, int(dynamic_size * 0.08)))
            gt_pixel = latlon_to_pixel(gt_lat, gt_lon, sat_bounds, w, h)
            cv2.drawMarker(img_sat, gt_pixel, (255, 0, 0), cv2.MARKER_STAR, dynamic_size, dynamic_thickness)
            print(f"GT center pixel: {gt_pixel}")

            # --- Optional: draw footprint ---
            if gt_alt is not None and gt_yaw is not None:
                footprint_pts = get_footprint_pixels(
                    gt_lat, gt_lon, gt_alt, gt_yaw, hfov, sat_bounds, w, h
                )
                cv2.polylines(img_sat, [footprint_pts], isClosed=True, color=(0, 255, 255), thickness=8)
                cv2.putText(img_sat, "UAV Footprint", (footprint_pts[0][0], footprint_pts[0][1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                print(f"Da ve Footprint tai do cao {gt_alt}m, goc xoay {gt_yaw} deg")

        except Exception as e:
            print(f"Loi ve GT/Footprint: {e}")

    # ==================== Highlight tam giac gan GT nhat ====================
    if gt_pixel is not None:
        min_dist = float('inf')
        for i, m in enumerate(matches):
            s = m['sat_row']
            cx, cy = get_triangle_centroid(s)
            d = ((cx - gt_pixel[0])**2 + (cy - gt_pixel[1])**2)**0.5
            if d < min_dist:
                min_dist = d
                best_i = i

        if best_i != -1:
            cx, cy = get_triangle_centroid(matches[best_i]['sat_row'])
            cv2.circle(img_sat, (cx, cy), 40, (0, 255, 255), 6)
            if sat_bounds is not None:
                dx_px = cx - gt_pixel[0]
                dy_px = cy - gt_pixel[1]
                distance_info = pixel_offset_to_meters(dx_px, dy_px, sat_bounds, w, h)
                print(
                    f"   -> Tam giac gan GT nhat: Top {best_i+1} "
                    f"(cach {min_dist:.0f} pixel ~ {distance_info['distance_m']:.1f} m | "
                    f"dx={distance_info['dx_m']:.1f} m, dy={distance_info['dy_m']:.1f} m)"
                )
                print(
                    f"   -> Satellite resolution x≈{distance_info['m_per_px_x']:.3f} m/px, "
                    f"y≈{distance_info['m_per_px_y']:.3f} m/px"
                )
            else:
                print(f"   -> Tam giac gan GT nhat: Top {best_i+1} (cach {min_dist:.0f} pixel)")

    # Màu sắc cho từng cặp khớp
    COLORS_BGR = [
        (220, 250,  250), (255, 140,  0), (50, 200,  50), (180,  50, 200), (0,  200, 220),   
        (220, 220,  0), (100,  50, 200), (139,  90,  43), (220, 130, 170), (0,  160, 130),  
        (255, 255, 255), (128, 128, 128), (255, 0, 255), (255, 255, 0), (0, 255, 255),    
        (255, 165, 255), (128, 0, 128), (0, 128, 128), (128, 128, 0), (255, 20, 147),
        (0, 255, 127), (255, 69, 0), (75, 0, 130), (255, 105, 180), (0, 191, 255),
        (25, 0, 0), (0, 25, 0), (0, 0, 25), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128),
        (255, 140, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255),
        (255, 215, 0), (255, 20, 147), (0, 255, 127), (255, 69, 0), (75, 0, 130),
        (255, 105, 180), (0, 191, 255), (25, 0, 0), (0, 25, 0), (0, 0, 25),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 128), (255, 140, 0), (255, 0, 0), (0, 255, 0),
        (0, 0, 255), (255, 255, 255), (255, 215, 0), (255, 20, 147), (0, 255, 127), 
        (255, 69, 0), (75, 0, 130), (255, 105, 180), (0, 191, 255), (25, 0, 0), 
        (0, 25, 0), (0, 0, 25), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128), (255, 140, 0),
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (255, 215, 0),
        (255, 20, 147), (0, 255, 127), (255, 69, 0), (75, 0, 130), (255, 105, 180),
        (0, 191, 255), (25, 0, 0), (0, 25, 0), (0, 0, 25), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 128), (255, 140, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 255), (255, 215, 0), (255, 20, 147), (0, 255, 127), (255, 69, 0)
    ]

    # Kích thước marker tỷ lệ với ảnh
    uav_marker = max(15, min(28, min(img_uav.shape[0], img_uav.shape[1]) // 18))
    sat_marker = max(18, min(80, min(img_sat.shape[0], img_sat.shape[1]) // 300))
    output_dir = os.path.dirname(sat_img_path) if os.path.dirname(sat_img_path) else "."

    if gt_pixel is not None and matches:
        points_preview_path = os.path.join(output_dir, 'matching_points_vs_gt.png')
        save_match_centroid_preview(
            img_sat_rgb=img_sat,
            matches=matches,
            output_path=points_preview_path,
            colors=COLORS_BGR,
            highlight_rank=(best_i + 1) if best_i != -1 else None,
        )
        print(f"   -> Da luu anh so sanh GT vs centroid: {points_preview_path}")

    # Vẽ lên ảnh UAV - bao gồm score
    for i, match in enumerate(matches):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        draw_match_triangle(img_uav, match['uav_row'], color, i + 1,
                            score=None,  # Không hiển thị score trên UAV để tránh rối mắt
                            thickness_tri=2, marker_size=uav_marker)

    # Vẽ lên ảnh Satellite - MỖI ĐỈNH và trọng tâm đều được đánh dấu + score
    for i, match in enumerate(matches):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        draw_match_triangle(img_sat, match['sat_row'], color, i + 1,
                            score=match['score'] if i < 20 else None,
                            thickness_tri=4, marker_size=sat_marker)

        if i < 10:
            # Giữ annotation đỉnh cho một số top đầu để tránh quá rối với top_k lớn.
            s = match['sat_row']
            for vi_idx, vi in enumerate(['vertex_0', 'vertex_1', 'vertex_2']):
                vx = int(s[f'{vi}_x'])
                vy = int(s[f'{vi}_y'])
                coord_text = f"({vx},{vy})"
                cv2.putText(img_sat, coord_text, (vx + 8, vy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # --- Figure chính ---
    fig = plt.figure(figsize=(22, 11))
    fig.patch.set_facecolor('#1a1a2e')

    # Title
    uav_label = "UAV Image (North-up, Yaw-corrected)" if uav_is_rotated else "UAV Image (Processed)"
    fig.suptitle(
        f"Ekeland Matching Results — Top {len(matches)} Pairs\n"
        f"Feature: 2 interior angles + 3 Ekeland angles (L1 distance)\n"
        f"UAV: {uav_label} | Satellite: Original",
        fontsize=13, fontweight='bold', color='white', y=0.98
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_uav)
    ax1.set_title(f"{uav_label}\n(Ekeland extracted on North-aligned image)",
                  fontsize=11, color='lightblue', pad=8)
    ax1.axis('off')

    # Đóng khung UAV với màu
    for spine in ax1.spines.values():
        spine.set_edgecolor('lightblue')
        spine.set_linewidth(2)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img_sat)
    ax2.set_title(
        "Satellite Image\n(Matched triangles + vertex markers)",
        fontsize=11, color='lightyellow', pad=8
    )
    ax2.axis('off')

    for spine in ax2.spines.values():
        spine.set_edgecolor('lightyellow')
        spine.set_linewidth(2)

    # Legend
    legend_patches = []
    for i in range(min(len(matches), len(COLORS_BGR), 20)):
        color_norm = tuple(c / 255.0 for c in COLORS_BGR[i])
        score = matches[i]['score']
        patch = mpatches.Patch(color=color_norm, label=f'Top {i+1} (score={score:.2f})')
        legend_patches.append(patch)

    fig.legend(handles=legend_patches, loc='lower center', ncol=min(len(matches), 5),
               fontsize=10, framealpha=0.8, facecolor='#2d2d4e', labelcolor='white',
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    # Lưu kết quả
    output_path = os.path.join(output_dir, 'matching_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n✓ Đã lưu visualization tại: {output_path}")
    plt.show()
    plt.close(fig)


# ==============================================================================
# PIPELINE ĐẦY ĐỦ: UAV → Xoay → Trích xuất → So khớp
# ==============================================================================
def run_full_pipeline(flight_csv,
                      uav_img_path, 
                      uav_csv_path_raw, 
                      sat_img_path, 
                      sat_csv_path,
                      top_k=1000, 
                      output_dir="/home/nguyenduytan/UAV_nonGPS/UAV_nonGPS_dataset/03/output"):
    
    # ĐẢM BẢO THƯ MỤC TỒN TẠI
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Đã tạo thư mục đầu ra: {output_dir}")

    print(f"Step 1: Xoay ảnh UAV về North-up...")
    rotated_path = os.path.join(output_dir, "uav_rotated.jpg")

    try:
        # Tạo ảnh xoay để visualization. Ưu tiên metadata từ flight_csv để đồng nhất
        # với pipeline trích xuất Ekeland ở bước sau.
        rotation_csv = flight_csv if (flight_csv and os.path.exists(flight_csv)) else uav_csv_path_raw

        _, img_rotated, meta = process_uav(
            img_path=uav_img_path,
            csv_path=rotation_csv
        )

        img_rotated.save(rotated_path, quality=95)
        print(f"  ✓ Đã lưu ảnh xoay thành công: {rotated_path}")
        
        # LƯU Ý: Tại đây bạn cần trích xuất lại đặc trưng từ ảnh ĐÃ XOAY
        # Nếu dùng CSV cũ (raw), tọa độ tam giác sẽ bị sai vị trí trên ảnh xoay.
        # uav_csv_path nên là đường dẫn tới file CSV đã được re-map tọa độ.
        uav_csv_to_match = uav_csv_path_raw 
        
    except Exception as e:
        print(f"  ✗ Lỗi trong quá trình xử lý/lưu ảnh: {e}")
        rotated_path = None
        uav_csv_to_match = uav_csv_path_raw

    print("\nStep 2: Tiến hành so khớp đặc trưng...")
    df = pd.read_csv(flight_csv)
    row = df[df['filename'] == "01_0002.JPG"].iloc[0]
    gt_lat, gt_lon = row['lat'], row['lon']

    bounds = load_satellite_bounds("satellite01.tif")   # sẽ in debug rất rõ

    uav_csv_rotated = complete_segmentation_demo_uav_with_rotation(
        model=model, device=device, uav_img_path=uav_img_path, uav_csv_path=flight_csv
    )

    # Nếu bước 1 chưa tạo được ảnh xoay, thử lại bằng đúng metadata đã dùng cho segmentation.
    if (not rotated_path) or (not os.path.exists(rotated_path)):
        try:
            _, img_rotated, _ = process_uav(
                img_path=uav_img_path,
                csv_path=flight_csv
            )
            img_rotated.save(rotated_path, quality=95)
            print(f"  ✓ Đã tạo lại ảnh xoay để hiển thị: {rotated_path}")
        except Exception as e:
            print(f"  ✗ Không thể tạo ảnh xoay để hiển thị: {e}")
            rotated_path = None
    
    # Sử dụng ảnh đã xoay để so khớp
    results = process_matching(
        uav_csv_path=uav_csv_rotated,
        sat_csv_path=sat_csv_path,
        uav_img_path=uav_img_path,        # Ảnh gốc (dự phòng)
        sat_img_path=sat_img_path,
        top_k=top_k,
        gt_lat=gt_lat,
        gt_lon=gt_lon,
        sat_bounds=bounds,
        uav_rotated_img_path=rotated_path # Ảnh đã xoay (ưu tiên hiển thị)
    )
    return results

class FeatureExtractor:
    """Extract feature vector from a CSV row using existing functional logic."""

    def __init__(self, include_shape_features=False, include_side_ratios=True, include_radius_ratio=True, include_elongation=True):
        self.include_shape_features = include_shape_features
        self.include_side_ratios = include_side_ratios
        self.include_radius_ratio = include_radius_ratio
        self.include_elongation = include_elongation

    def extract_from_row(self, row):
        return extract_features_from_row(
            row,
            include_shape_features=self.include_shape_features,
            include_side_ratios=self.include_side_ratios,
            include_radius_ratio=self.include_radius_ratio,
            include_elongation=self.include_elongation,
        )


class DistanceMetric:
    """Distance scorer facade using existing compute_feature_distance()."""

    def __init__(self, metric='complex', vm_kappa=4.0, angle_feature_dims=5, linear_weight=1.0):
        self.metric = metric
        self.vm_kappa = vm_kappa
        self.angle_feature_dims = angle_feature_dims
        self.linear_weight = linear_weight

    def score(self, feat_a, feat_b):
        return compute_feature_distance(
            feat_a,
            feat_b,
            metric=self.metric,
            vm_kappa=self.vm_kappa,
            angle_feature_dims=self.angle_feature_dims,
            linear_weight=self.linear_weight,
        )

class Matcher:
    """Brute-force matcher facade preserving the current matching logic."""

    def __init__(self, extractor=None, distance_metric=None):
        self.extractor = extractor if extractor is not None else FeatureExtractor()
        self.distance_metric = distance_metric if distance_metric is not None else DistanceMetric()

    def match_dataframe(self, df_uav, df_sat, top_k=1000):
        uav_features = [self.extractor.extract_from_row(row) for _, row in df_uav.iterrows()]
        sat_features = [self.extractor.extract_from_row(row) for _, row in df_sat.iterrows()]
        if top_k <= 0:
            return []

        top_heap = []
        for idx_u, feat_u in enumerate(uav_features):
            for idx_s, feat_s in enumerate(sat_features):
                score = self.distance_metric.score(feat_u, feat_s)
                candidate = (-score, idx_u, idx_s)
                if len(top_heap) < top_k:
                    heapq.heappush(top_heap, candidate)
                elif score < -top_heap[0][0]:
                    heapq.heapreplace(top_heap, candidate)

        top_candidates = sorted(top_heap, key=lambda x: -x[0])  # ascending score
        return [
            {
                'score': -neg_score,
                'uav_idx': idx_u,
                'sat_idx': idx_s,
                'uav_row': df_uav.iloc[idx_u],
                'sat_row': df_sat.iloc[idx_s],
            }
            for neg_score, idx_u, idx_s in top_candidates
        ]


class MatchVisualizer:
    """Visualization facade around existing visualize_matches()."""

    def render(self, uav_img_path, sat_img_path, matches, **kwargs):
        return visualize_matches(uav_img_path, sat_img_path, matches, **kwargs)


class MatchingService:
    """Service facade around existing process_matching() for OOP orchestration."""

    def run(self, uav_csv_path, sat_csv_path, uav_img_path, sat_img_path, **kwargs):
        return process_matching(
            uav_csv_path=uav_csv_path,
            sat_csv_path=sat_csv_path,
            uav_img_path=uav_img_path,
            sat_img_path=sat_img_path,
            **kwargs,
        )
