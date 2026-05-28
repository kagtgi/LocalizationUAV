import os
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon
from scipy.spatial import Delaunay
from .contours import extract_contours_from_mask, contour_to_polygon, filter_large_polygons, visualize_ekeland
from torchvision import transforms
from ..geometry.ekeland import compute_expansion_ekeland_for_all_triangles
from ..io.export import export_expansion_ekeland_to_csv

def complete_segmentation_demo(model, data_loader, device, num_samples=2, 
                                method='marching_squares', max_search_radius=100.0,
                                image_path=None):
    """
    Segmentation pipeline với Ekeland cho ảnh cụ thể hoặc ảnh từ DataLoader
    """
    model.eval()

    print(f"Running Segmentation Demo with Ekeland")
    print(f"Method: {method.upper()}")
    print(f"Ekeland search radius: {max_search_radius}px")
    
    # Chuẩn bị danh sách batch để xử lý
    batches_to_process = []
    
    if image_path is not None:
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} does not exist.")
            return
        
        print(f"Processing specific image: {image_path}")
        # Tải và tiền xử lý ảnh (giống với transform trong Dataset)
        img_pil = Image.open(image_path).convert("RGB")
        
        # Transform chuẩn của model (ResNet50 FPN thường dùng các giá trị này)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = preprocess(img_pil).to(device)
        # Giả lập cấu trúc batch của DataLoader: (danh sách ảnh, danh sách target)
        batches_to_process = [( [img_tensor], [None] )]
        num_samples = 1 
    else:
        batches_to_process = data_loader

    print("=" * 70)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(batches_to_process):
            if batch_idx >= num_samples:
                break

            # Nếu từ DataLoader, images có thể là tuple cần chuyển sang device
            if image_path is None:
                images = [img.to(device) for img in images]
            
            predictions = model(images)

            for i, (image, prediction, target) in enumerate(zip(images, predictions, targets)):
                # 1. Denormalize image để hiển thị
                image_cpu = image.cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_denorm = image_cpu * std + mean
                image_denorm = torch.clamp(image_denorm, 0, 1)
                image_np = image_denorm.permute(1, 2, 0).numpy()
                image_np = (image_np * 255).astype(np.uint8)

                # 2. Confidence filtering
                scores = prediction['scores'].cpu().numpy()
                keep = scores >= 0.5

                if not keep.any():
                    print("   No confident predictions found")
                    continue

                pred_masks = prediction['masks'][keep].cpu().numpy()
                pred_scores = prediction['scores'][keep].cpu().numpy()
                pred_boxes = prediction['boxes'][keep].cpu().numpy()

                img_label = f"Image {batch_idx + 1}-{i + 1}" if image_path is None else os.path.basename(image_path)
                print(f"\n   Processing: {img_label}")
                print(f"   Found {len(pred_masks)} confident predictions")

                # 3. Extract contours
                combined_contours = []
                combined_polygons = []
                combined_mask = np.zeros_like(pred_masks[0][0], dtype=np.uint8)

                for j, (mask, score, box) in enumerate(zip(pred_masks, pred_scores, pred_boxes)):
                    binary_mask = mask[0] if len(mask.shape) == 3 else mask
                    
                    processing_mask = (binary_mask > 0.5).astype(np.uint8) if method == 'opencv' else binary_mask
                    vis_mask_part = (binary_mask > 0.5).astype(np.uint8)
                    combined_mask = np.logical_or(combined_mask, vis_mask_part)

                    contours = extract_contours_from_mask(processing_mask, min_area=50, method=method)
                    polygons = [contour_to_polygon(c) for c in contours]
                    
                    if not contours: continue
                    combined_contours.extend(contours)
                    combined_polygons.extend(polygons)

                # 4. Filter large polygons (biên ảnh/nhiễu)
                combined_contours, combined_polygons = filter_large_polygons(
                    combined_contours, combined_polygons, image_np.shape, max_size_ratio=0.20
                )

                if len(combined_polygons) == 0:
                    print("   No valid polygons after filtering")
                    continue

                # 5. Create Shapely obstacles
                all_obstacle_polygons = []
                for polygon in combined_polygons:
                    try:
                        shapely_poly = Polygon(polygon)
                        if shapely_poly.is_valid:
                            all_obstacle_polygons.append(shapely_poly)
                    except: pass

                # 6. Delaunay Triangulation & EKELAND
                polygon_triangulations = []
                all_expansion_results = []

                for poly_idx, polygon in enumerate(combined_polygons):
                    if len(polygon) < 3:
                        polygon_triangulations.append(None)
                        continue
                    try:
                        points = np.array(polygon, dtype=np.float64)
                        tri = Delaunay(points)
                        polygon_triangulations.append({'points': points, 'triangles': tri.simplices, 'tri_object': tri})
                        
                        # Remove current polygon from obstacles
                        other_obstacles = [obs for j, obs in enumerate(all_obstacle_polygons) if j != poly_idx]
                        
                        expansion_results = compute_expansion_ekeland_for_all_triangles(
                            tri, all_polygons_in_image=other_obstacles, max_search_radius=max_search_radius
                        )
                        all_expansion_results.extend(expansion_results)
                    except Exception as e:
                        print(f"      WARNING: Failed for polygon {poly_idx + 1}: {e}")
                        polygon_triangulations.append(None)

                # 7. Export CSV
                if all_expansion_results:
                    if image_path:
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        csv_filename = f"ekeland_{base_name}.csv"
                    else:
                        csv_filename = f"ekeland_img_{batch_idx}_{i}.csv"
                    
                    export_expansion_ekeland_to_csv(all_expansion_results, csv_filename)
                    print(f"   ✓ Exported {len(all_expansion_results)} results to {csv_filename}")

                # 8. Visualize
                if combined_contours and combined_polygons:
                    combined_mask_vis = (combined_mask.astype(np.uint8)) * 255
                    
                    yaw_val = 162.5 
                    custom_title = f"UAV Ekeland (North-up, yaw={yaw_val}° corrected)"
                    
                    visualize_ekeland(
                        image=image_np, 
                        mask=combined_mask_vis, 
                        contours=combined_contours, 
                        polygons=combined_polygons,
                        triangulations=polygon_triangulations, 
                        expansion_results=all_expansion_results,
                        title=custom_title
                    )

    print("\n" + "=" * 70)
    print("Complete segmentation demo finished!")


class SegmentationPipeline:
    """Class-based wrapper for the segmentation demo pipeline."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def run(self, data_loader=None, num_samples=2, method='marching_squares', max_search_radius=100.0, image_path=None):
        return complete_segmentation_demo(
            model=self.model,
            data_loader=data_loader,
            device=self.device,
            num_samples=num_samples,
            method=method,
            max_search_radius=max_search_radius,
            image_path=image_path,
        )
