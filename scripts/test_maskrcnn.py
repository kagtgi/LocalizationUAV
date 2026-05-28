import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from localization.segmentation.model import get_model

# ──────────────────────────────────────────────────────────────────────────────
# HƯỚNG DẪN CHẠY:
#   python3 test_maskrcnn.py \
#       --data-root .. --gf7-bands 3 \
#       --checkpoint checkpoints/maskrcnn_gf7_3bands/best_model.pth \
#       --score-thresh 0.5 \
#       --save-masks
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    default_data_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Test Mask R-CNN on GF-7 building datasets.")

    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--gf7-bands", type=int, default=3, choices=[3, 4])
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Nên để batch=1 cho test để dễ dàng lưu tên file và xử lý kích thước ảnh khác nhau.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-instance-area", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--checkpoint", type=Path, required=True, 
                        help="Đường dẫn tới file .pth (ví dụ: best_model.pth)")
    parser.add_argument("--score-thresh", type=float, default=0.5, 
                        help="Ngưỡng tự tin (confidence score) để giữ lại mask.")
    
    # Tùy chọn lưu kết quả
    parser.add_argument("--output-dir", type=Path, default=Path("test_results"))
    parser.add_argument("--save-vis", action="store_true", default=True,
                        help="Lưu ảnh overlay so sánh Ground Truth và Prediction.")
    parser.add_argument("--save-masks", action="store_true",
                        help="Lưu raw binary masks (.tif) của prediction.")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data utilities (Tái sử dụng logic từ train)
# ──────────────────────────────────────────────────────────────────────────────

def resolve_gf7_root(data_root, bands):
    return data_root / ("GF-7 Building (3Bands)" if bands == 3 else "GF-7 Building (4Bands)")

def collect_pairs(split_root):
    image_dir = split_root / "image"
    label_dir = split_root / "label"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing split folders: {image_dir} or {label_dir}")
    image_paths = sorted(image_dir.glob("*.tif"))
    pairs = []
    for img_path in image_paths:
        lbl_path = label_dir / img_path.name
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            print(f"[WARN] Missing label for {img_path.name}")
    return pairs

def adapt_channels(image, in_channels):
    if image.ndim == 2: image = image[..., None]
    if image.shape[0] <= 8 and image.shape[-1] > 8: image = np.transpose(image, (1, 2, 0))
    c = image.shape[-1]
    if c == in_channels: return np.ascontiguousarray(image)
    if c > in_channels: return np.ascontiguousarray(image[..., :in_channels])
    if c == 1:
        image = np.repeat(image, repeats=min(in_channels, 3), axis=-1)
        c = image.shape[-1]
    if c < in_channels:
        pad = np.repeat(image[..., -1:], repeats=in_channels - c, axis=-1)
        image = np.concatenate([image, pad], axis=-1)
    return np.ascontiguousarray(image)

def normalize_image(image):
    if np.issubdtype(image.dtype, np.integer):
        denom = float(np.iinfo(image.dtype).max)
    else:
        mv = float(np.nanmax(image))
        denom = 1.0 if mv <= 1.0 else (255.0 if mv <= 255.0 else mv)
    return np.clip(image.astype(np.float32) / max(denom, 1.0), 0.0, 1.0)

def to_binary_mask(mask):
    if mask.ndim == 3: mask = mask[..., 0]
    return (mask > 127).astype(np.uint8)

def build_target_from_mask(mask, min_instance_area):
    binary = to_binary_mask(mask)
    num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes, labels, masks = [], [], []
    for lid in range(1, num_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_instance_area: continue
        x, y = int(stats[lid, cv2.CC_STAT_LEFT]), int(stats[lid, cv2.CC_STAT_TOP])
        w, h = int(stats[lid, cv2.CC_STAT_WIDTH]), int(stats[lid, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0: continue
        boxes.append([x, y, x + w, y + h])
        labels.append(1)
        masks.append(labeled == lid)

    if masks:
        return {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks, 0), dtype=torch.uint8)
        }
    else:
        h_img, w_img = binary.shape
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, h_img, w_img), dtype=torch.uint8)
        }

class GF7TestDataset(Dataset):
    def __init__(self, pairs, in_channels, min_instance_area=20):
        self.pairs = pairs
        self.in_channels = in_channels
        self.min_instance_area = min_instance_area

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, lbl_path = self.pairs[index]
        image = tifffile.imread(str(img_path))
        mask = tifffile.imread(str(lbl_path))
        
        image_proc = adapt_channels(image, self.in_channels)
        image_proc = normalize_image(image_proc)
        image_tensor = torch.from_numpy(image_proc).permute(2, 0, 1).contiguous()
        
        target = build_target_from_mask(mask, self.min_instance_area)
        
        # Trả về thêm img_path để biết tên file khi lưu
        return image_tensor, target, str(img_path)

def collate_fn_test(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


# ──────────────────────────────────────────────────────────────────────────────
# Logic Test và Trực quan hóa
# ──────────────────────────────────────────────────────────────────────────────

def make_overlay(base, mask, color, alpha=0.45):
    out = base.copy()
    where = mask > 0.5
    for c, v in enumerate(color):
        out[..., c] = np.where(where, base[..., c] * (1 - alpha) + v * alpha, base[..., c])
    return out

@torch.no_grad()
def run_test(model, data_loader, device, args):
    model.eval()
    
    vis_dir = args.output_dir / "visualizations"
    mask_dir = args.output_dir / "predicted_masks"
    
    if args.save_vis: vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks: mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Bắt đầu test trên {len(data_loader.dataset)} ảnh...")
    
    # Các biến để tích lũy TP, FP, FN cho toàn bộ tập dataset
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for images, targets, paths in tqdm(data_loader, desc="Testing"):
        images_dev = [img.to(device) for img in images]
        
        use_amp = (device.type == "cuda")
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images_dev)
            
        for i in range(len(images)):
            img_tensor = images[i]
            target = targets[i]
            output = outputs[i]
            img_name = Path(paths[i]).name
            
            # --- 1. Lấy Prediction Mask ---
            pred_masks_all = output["masks"].cpu().squeeze(1)
            pred_scores = output["scores"].cpu().numpy()
            keep_idx = np.where(pred_scores >= args.score_thresh)[0]
            
            if len(keep_idx) > 0:
                pred_masks_kept = pred_masks_all[keep_idx] > 0.5
                pred_combined = pred_masks_kept.float().max(dim=0).values.numpy()
            else:
                pred_combined = np.zeros(img_tensor.shape[1:], dtype=np.float32)

            # --- 2. Lấy Ground Truth Mask ---
            gt_masks = target["masks"].numpy()
            gt_combined = gt_masks.max(axis=0).astype(np.float32) if len(gt_masks) > 0 else np.zeros_like(pred_combined)

            # --- 3. Tính toán TP, FP, FN cho ảnh này ---
            pred_bool = pred_combined > 0.5
            gt_bool = gt_combined > 0.5
            
            tp = np.logical_and(pred_bool, gt_bool).sum()
            fp = np.logical_and(pred_bool, ~gt_bool).sum()
            fn = np.logical_and(~pred_bool, gt_bool).sum()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # --- 4. Lưu Masks và Visualizations (giữ nguyên như cũ) ---
            if args.save_masks:
                mask_save_path = mask_dir / img_name
                tifffile.imwrite(str(mask_save_path), (pred_combined * 255).astype(np.uint8))

            if args.save_vis:
                # ... (Đoạn code visualization matplotlib giữ nguyên như phần trước) ...
                pass

    # --- TÍNH TOÁN METRICS TỔNG THỂ SAU KHI CHẠY XONG DATASET ---
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (PIXEL-LEVEL)")
    print("="*50)
    print(f"  Threshold  : {args.score_thresh}")
    print(f"  Precision  : {precision * 100:.2f} %")
    print(f"  Recall     : {recall * 100:.2f} %")
    print(f"  F1-Score   : {f1_score * 100:.2f} %")
    print(f"  IoU        : {iou * 100:.2f} %")
    print("="*50)
    print(f"[DONE] Hoàn tất! Kết quả được lưu tại: {args.output_dir}")

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dùng thiết bị: {device}")

    # 1. Dataset & DataLoader
    data_root = args.data_root.resolve()
    gf7_root = resolve_gf7_root(data_root, args.gf7_bands)
    
    # Sửa "Test" thành tên thư mục chứa tập test của bạn (có thể là "Test" hoặc "Val")
    test_split_path = gf7_root / "Test" 
    if not test_split_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục Test tại: {test_split_path}")

    test_pairs = collect_pairs(test_split_path)
    in_channels = 3 if args.gf7_bands == 3 else 4
    
    test_dataset = GF7TestDataset(test_pairs, in_channels, min_instance_area=args.min_instance_area)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=collate_fn_test
    )
    print(f"[INFO] Tìm thấy {len(test_dataset)} ảnh trong tập Test.")

    # 2. Khởi tạo Model
    model = get_model(num_classes=2, pretrained=False, in_channels=in_channels)
    
    # 3. Load Checkpoint (Ưu tiên EMA nếu có)
    ckpt_path = args.checkpoint.resolve()
    print(f"[INFO] Loading checkpoint từ: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
        print("[INFO] Đã tìm thấy trọng số EMA! Tiến hành load EMA model (Tốt cho Test).")
        model.load_state_dict(checkpoint["ema_state_dict"])
    elif "model_state_dict" in checkpoint:
        print("[INFO] Đang load trọng số Model chuẩn...")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Hỗ trợ load state_dict thuần túy
        model.load_state_dict(checkpoint)
        
    model.to(device)

    # 4. Bắt đầu Test
    run_test(model, test_loader, device, args)

if __name__ == "__main__":
    main()
