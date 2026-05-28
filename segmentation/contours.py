import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure

def extract_contours_from_mask(mask, min_area=100, method='marching_squares'):
    """Extract contours from mask"""
    valid_contours = []

    if method == 'opencv':
        if mask.max() <= 1:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                valid_contours.append(cnt)

    elif method == 'marching_squares':
        if mask.dtype == np.uint8:
            # Support both binary uint8 masks in [0,1] and image-like masks in [0,255].
            if mask.max() <= 1:
                mask_float = mask.astype(np.float32)
            else:
                mask_float = mask.astype(np.float32) / 255.0
            level = 0.5
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

def contour_to_polygon(contour, epsilon_factor=0.02):
    """Convert contour to simplified polygon"""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon.reshape(-1, 2).tolist()


def contour_to_polygon_dynamic(contour, image_shape, min_epsilon_factor=0.002, max_epsilon_factor=0.02):
    """
    Convert contour to polygon with dynamic simplification.
    Small contours keep more detail (smaller epsilon), large contours simplify more.
    """
    if len(image_shape) == 3:
        img_height, img_width = image_shape[:2]
    else:
        img_height, img_width = image_shape

    image_area = max(float(img_height * img_width), 1.0)
    area = max(float(cv2.contourArea(contour)), 1.0)
    area_ratio = np.clip(area / image_area, 0.0, 1.0)
    # Interpolate epsilon factor in log-like way to preserve small shapes.
    t = np.sqrt(area_ratio)
    epsilon_factor = min_epsilon_factor + (max_epsilon_factor - min_epsilon_factor) * t
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    return polygon.reshape(-1, 2).tolist()

def filter_large_polygons(contours, polygons, image_shape, max_size_ratio=0.20):
    """Filter out polygons that are too large"""
    if len(image_shape) == 3:
        img_height, img_width = image_shape[:2]
    else:
        img_height, img_width = image_shape

    image_area = img_height * img_width
    max_allowed_area = image_area * max_size_ratio

    filtered_contours = []
    filtered_polygons = []
    removed_count = 0

    for contour, polygon in zip(contours, polygons):
        polygon_area = cv2.contourArea(contour)
        if polygon_area >= max_allowed_area:
            removed_count += 1
        else:
            filtered_contours.append(contour)
            filtered_polygons.append(polygon)

    if removed_count > 0:
        print(f"   Filtered out {removed_count} large polygon(s)")

    return filtered_contours, filtered_polygons


def filter_large_polygons_dynamic(contours, polygons, image_shape, quantile=0.995, fallback_max_size_ratio=0.95):
    """
    Dynamic filter for very large polygons based on contour area distribution.
    Keeps almost all candidates and removes only extreme outliers.
    """
    if not contours:
        return contours, polygons

    if len(image_shape) == 3:
        img_height, img_width = image_shape[:2]
    else:
        img_height, img_width = image_shape
    image_area = float(max(img_height * img_width, 1))
    hard_cap = image_area * fallback_max_size_ratio

    areas = np.array([max(float(cv2.contourArea(c)), 0.0) for c in contours], dtype=np.float64)
    if len(areas) >= 4:
        q_cap = float(np.quantile(areas, quantile))
        max_allowed_area = min(max(q_cap, 1.0), hard_cap)
    else:
        max_allowed_area = hard_cap

    filtered_contours = []
    filtered_polygons = []
    removed_count = 0
    for contour, polygon, area in zip(contours, polygons, areas):
        if area > max_allowed_area:
            removed_count += 1
        else:
            filtered_contours.append(contour)
            filtered_polygons.append(polygon)

    if removed_count > 0:
        print(f"   Dynamically filtered out {removed_count} extreme large polygon(s)")
    return filtered_contours, filtered_polygons

def get_ekeland_color(ekeland_angle, colormap='RdYlGn'):
    """Map Ekeland angle to color (0° = red, 180° = green)"""
    normalized = np.clip(ekeland_angle / 180.0, 0, 1)
    cmap = plt.get_cmap(colormap)
    rgba = cmap(normalized)
    bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
    return bgr

def visualize_ekeland(
        image, 
        mask, 
        contours,
        polygons, 
        triangulations,
        expansion_results, 
        title="Ekeland", 
        max_ekeland_triangles=None):
    """Visualize with 6 panels"""
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if image.ndim == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image_color = cv2.cvtColor(image.squeeze(-1), cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    h, w = image_color.shape[:2]

    # Panel 1: Original Image
    axes[0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Binary Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Binary Mask\n{len(polygons)} buildings', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Contours
    contour_img = image_color.copy()
    for contour in contours:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cnt_int = contour.astype(np.int32)
        cv2.drawContours(contour_img, [cnt_int], -1, color, 2)

    axes[2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Contours\n{len(contours)} detected', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Panel 4: Polygons
    polygon_img = image_color.copy()
    for polygon in polygons:
        polygon_np = np.array(polygon, dtype=np.int32)
        cv2.polylines(polygon_img, [polygon_np], True, (0, 255, 0), 2)
        for (x, y) in polygon:
            cv2.circle(polygon_img, (int(x), int(y)), 3, (255, 0, 0), -1)

    axes[3].imshow(cv2.cvtColor(polygon_img, cv2.COLOR_BGR2RGB))
    total_vertices = sum(len(p) for p in polygons)
    axes[3].set_title(f'Polygons\n{total_vertices} vertices', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    # Panel 5: Delaunay Triangulation
    triangle_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    total_triangles = 0
    np.random.seed(42)

    for poly_idx, tri_data in enumerate(triangulations):
        if tri_data is None:
            continue

        points = tri_data['points']
        triangles = tri_data['triangles']
        total_triangles += len(triangles)

        hue = int((poly_idx * 68.75) % 180)
        color_hsv = np.array([[[hue, 200, 200]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = tuple(int(c) for c in color_bgr)

        for simplex in triangles:
            triangle_pts = points[simplex].astype(np.int32)
            overlay = triangle_img.copy()
            cv2.fillPoly(overlay, [triangle_pts], color)
            cv2.addWeighted(overlay, 0.3, triangle_img, 0.7, 0, triangle_img)

        for simplex in triangles:
            triangle_pts = points[simplex].astype(np.int32)
            cv2.polylines(triangle_img, [triangle_pts], True, (0, 0, 0), 1)

    axes[4].imshow(cv2.cvtColor(triangle_img, cv2.COLOR_BGR2RGB))
    axes[4].set_title(f'Delaunay Triangulation\n{total_triangles} triangles',
                     fontsize=12, fontweight='bold')
    axes[4].axis('off')

    # Panel 6: Ekeland Angles with Cones
    # THAY ĐỔI 1: Sử dụng ảnh UAV gốc (image_color) làm nền thay vì nền trắng (ones * 255)
    ekeland_img = image_color.copy()

    # Vẽ nền polygons bán trong suốt để làm nổi bật cấu trúc khối tòa nhà trên ảnh UAV
    overlay_poly = ekeland_img.copy()
    for polygon in polygons:
        polygon_pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay_poly, [polygon_pts], (230, 230, 230))
        cv2.polylines(ekeland_img, [polygon_pts], True, (100, 100, 100), 2)
    # Trộn nhẹ nền polygon màu xám lên ảnh gốc
    cv2.addWeighted(overlay_poly, 0.2, ekeland_img, 0.8, 0, ekeland_img)

    # 1. Tính toán thống kê góc cho TẤT CẢ các tam giác (giữ nguyên để Title chính xác)
    all_ekeland_angles = []
    for result in expansion_results:
        for ek_data in result.get('ekeland_full_results', {}).values():
            all_ekeland_angles.append(ek_data['ekeland_angle'])

    # 2. THAY ĐỔI 2: Lấy mẫu NGẪU NHIÊN tam giác cho mỗi lần vẽ
    results_to_draw = expansion_results
    if max_ekeland_triangles is not None and len(expansion_results) > max_ekeland_triangles:
        # Sử dụng indices ngẫu nhiên không lặp lại từ numpy
        random_indices = np.random.choice(len(expansion_results), size=max_ekeland_triangles, replace=False)
        results_to_draw = [expansion_results[idx] for idx in random_indices]

    cone_radius = 50

    # 3. Vẽ Cone cho các tam giác ngẫu nhiên đã chọn đè lên nền ảnh UAV
    for result in results_to_draw:
        ekeland_full = result.get('ekeland_full_results', {})
        seed_coords = np.array(result['seed_coordinates'])
        
        for vertex_idx, ek_data in ekeland_full.items():
            ekeland_angle = ek_data['ekeland_angle']
            best_axis = ek_data['best_axis']
            axis_name = ek_data['axis_name']
            
            # Tìm tọa độ
            vertex_pos_idx = result['seed_vertices'].index(vertex_idx) if vertex_idx in result['seed_vertices'] else -1
            if vertex_pos_idx == -1:
                continue
            
            position = seed_coords[vertex_pos_idx]
            x, y = position
            pt = (int(x), int(y))
            
            color = get_ekeland_color(ekeland_angle)
            
            # Vẽ cone (fan shape) theo bisector
            if ekeland_angle > 1.0:
                bisector_angle_deg = np.degrees(np.arctan2(best_axis[1], best_axis[0]))
                half_angle = ekeland_angle / 2.0
                
                angle_start = bisector_angle_deg - half_angle
                angle_end = bisector_angle_deg + half_angle
                
                n_points = max(int(ekeland_angle / 5), 10)
                arc_angles = np.linspace(np.radians(angle_start), np.radians(angle_end), n_points)
                
                arc_points = [pt]
                for angle in arc_angles:
                    end_x = int(x + cone_radius * np.cos(angle))
                    end_y = int(y + cone_radius * np.sin(angle))
                    arc_points.append((end_x, end_y))
                arc_points.append(pt)
                
                arc_points_np = np.array(arc_points, dtype=np.int32)
                overlay = ekeland_img.copy()
                cv2.fillPoly(overlay, [arc_points_np], color)
                # Tăng alpha trộn từ 0.4 lên 0.55 giúp hình quạt đỏ/xanh hiển thị rõ nét hơn trên nền ảnh UAV phức tạp
                cv2.addWeighted(overlay, 0.55, ekeland_img, 0.45, 0, ekeland_img)
            
            # Vẽ đỉnh
            cv2.circle(ekeland_img, pt, 6, color, -1)
            cv2.circle(ekeland_img, pt, 6, (0, 0, 0), 1)
            
            # Label số đo góc (Thêm viền text hoặc đổi màu trắng nếu nền ảnh UAV quá tối)
            # Ở đây ta dùng text màu đen, bạn có thể đổi thành (255, 255, 255) nếu muốn nổi bật trên nền ảnh UAV
            cv2.putText(ekeland_img, f"{ekeland_angle:.0f}", (int(x)+10, int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    axes[5].imshow(cv2.cvtColor(ekeland_img, cv2.COLOR_BGR2RGB))
    
    if all_ekeland_angles:
        mean_ek = np.mean(all_ekeland_angles)
        max_ek = np.max(all_ekeland_angles)
        min_ek = np.min(all_ekeland_angles)
        axes[5].set_title(
            f'Ekeland (Max Free Cone)\n'
            f'Mean: {mean_ek:.1f}° | Range: [{min_ek:.1f}°, {max_ek:.1f}°]',
            fontsize=12, fontweight='bold'
        )
    else:
        axes[5].set_title('Ekeland', fontsize=12, fontweight='bold')
    
    axes[5].axis('off')

    # Add legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0,0),1,1, fc=np.array(get_ekeland_color(150))/255, label='Large (>120°)'),
        Rectangle((0,0),1,1, fc=np.array(get_ekeland_color(90))/255, label='Medium (60-120°)'),
        Rectangle((0,0),1,1, fc=np.array(get_ekeland_color(30))/255, label='Small (<60°)')
    ]
    axes[5].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    import os
    output_filename = "UAV_6_panels_visualization.jpg"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved 6-step visualization: {output_filename}")
    
    plt.show()
