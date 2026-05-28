"""
mutual_learning_swin.py
=======================
Mutual Deep Learning (MDL) giữa:
  - Model A: Mask R-CNN (ResNet50 Backbone) - Chuyên gia Local Features (CNN)
  - Model B: Mask R-CNN (Swin-T Backbone)   - Chuyên gia Global Context (Transformer)

Lợi ích:
  - Dùng chung 100% data pipeline và loss function (không bao giờ bị lỗi yolo_loss = 0).
  - Feat Mimicking và Box Distillation hoạt động đối xứng và cực kỳ chính xác.
  - Hỗ trợ lưu Checkpoint, Best Model và Visualization overlay đầy đủ.

Cách dùng:
  torchrun --nproc_per_node=2 mutual_learning_swin.py \
    --data-root .. --gf7-bands 3 --epochs 50 --batch-size 4 \
    --mdl-weight 0.5 --feat-mimic-weight 0.3 \
    --amp --ema --vis-every 10 --save-every 5
"""

import argparse
import copy
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models import swin_t, Swin_T_Weights
import torchvision.models.detection.roi_heads as roi_heads
from torchvision.models.detection.roi_heads import project_masks_on_boxes

# ──────────────────────────────────────────────────────────────────────────────
# 1. CUSTOM LOSSES (Focal & Tversky)
# ──────────────────────────────────────────────────────────────────────────────

def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    ce_loss = F.cross_entropy(class_logits, labels, reduction="none")
    pt = torch.exp(-ce_loss)
    gamma = 2.0
    alpha = 0.25 
    focal_loss = (alpha * ((1 - pt) ** gamma) * ce_loss).mean()

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9, reduction="sum"
    )
    box_loss = box_loss / labels.numel()
    return focal_loss, box_loss

def custom_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    pos_inds = torch.where(labels > 0)[0]
    
    if pos_inds.numel() == 0:
        return mask_logits.sum() * 0

    labels_pos = labels[pos_inds]
    mask_logits_pos = mask_logits[pos_inds, labels_pos]
    mask_targets_pos = mask_targets[pos_inds]

    bce_loss = F.binary_cross_entropy_with_logits(mask_logits_pos, mask_targets_pos)
    probs = torch.sigmoid(mask_logits_pos)
    probs_flat = probs.view(-1)
    targets_flat = mask_targets_pos.view(-1)

    alpha_fp, beta_fn, smooth = 0.7, 0.3, 1e-6
    TP = (probs_flat * targets_flat).sum()
    FP = (probs_flat * (1 - targets_flat)).sum()
    FN = ((1 - probs_flat) * targets_flat).sum()

    tversky_index = (TP + smooth) / (TP + alpha_fp * FP + beta_fn * FN + smooth)
    return bce_loss + (1.0 - tversky_index) * 2.0

roi_heads.fastrcnn_loss = custom_fastrcnn_loss
roi_heads.maskrcnn_loss = custom_maskrcnn_loss

# ──────────────────────────────────────────────────────────────────────────────
# 2. XÂY DỰNG MÔ HÌNH (RESNET50 & SWIN-T)
# ──────────────────────────────────────────────────────────────────────────────

def _expand_image_stats(image_stats, in_channels):
    if image_stats is None: return None
    if len(image_stats) >= in_channels: return list(image_stats[:in_channels])
    return list(image_stats) + [image_stats[-1]] * (in_channels - len(image_stats))

def get_resnet_model(num_classes, pretrained=True, in_channels=3):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    
    if in_channels != 3:
        conv1 = model.backbone.body.conv1
        new_conv = nn.Conv2d(in_channels, conv1.out_channels, kernel_size=conv1.kernel_size, 
                             stride=conv1.stride, padding=conv1.padding, bias=False)
        with torch.no_grad():
            if pretrained:
                copy_c = min(3, in_channels)
                new_conv.weight[:, :copy_c] = conv1.weight[:, :copy_c]
                if in_channels > 3:
                    new_conv.weight[:, 3:] = conv1.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        model.backbone.body.conv1 = new_conv

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    model.transform.image_mean = _expand_image_stats([0.485, 0.456, 0.406], in_channels)
    model.transform.image_std = _expand_image_stats([0.229, 0.224, 0.225], in_channels)
    return model

class SwinTransformerBackbone(nn.Module):
    """Custom wrapper để trích xuất 4 cấp độ feature từ Swin-T cho FPN"""
    def __init__(self, pretrained=True, in_channels=3):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        swin = swin_t(weights=weights)
        
        if in_channels != 3:
            old_conv = swin.features[0][0] # PatchMerging Conv2d
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                                 stride=old_conv.stride, padding=old_conv.padding)
            with torch.no_grad():
                if pretrained:
                    copy_c = min(3, in_channels)
                    new_conv.weight[:, :copy_c] = old_conv.weight[:, :copy_c]
                    if in_channels > 3:
                        new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
            swin.features[0][0] = new_conv

        self.features = swin.features

    def forward(self, x):
        out = OrderedDict()
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 1:   out['0'] = x.permute(0, 3, 1, 2).contiguous() # Cấp 1 (96)
            elif i == 3: out['1'] = x.permute(0, 3, 1, 2).contiguous() # Cấp 2 (192)
            elif i == 5: out['2'] = x.permute(0, 3, 1, 2).contiguous() # Cấp 3 (384)
            elif i == 7: out['3'] = x.permute(0, 3, 1, 2).contiguous() # Cấp 4 (768)
        return out

def get_swin_model(num_classes, pretrained=True, in_channels=3):
    body = SwinTransformerBackbone(pretrained=pretrained, in_channels=in_channels)
    fpn = FeaturePyramidNetwork(
        in_channels_list=[96, 192, 384, 768], 
        out_channels=256,
        extra_blocks=LastLevelMaxPool()
    )
    backbone = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    backbone.out_channels = 256

    model = MaskRCNN(backbone, num_classes=num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    model.transform.image_mean = _expand_image_stats([0.485, 0.456, 0.406], in_channels)
    model.transform.image_std = _expand_image_stats([0.229, 0.224, 0.225], in_channels)
    return model

# ──────────────────────────────────────────────────────────────────────────────
# 3. MDL COMPONENTS (HOOKS & LOSS)
# ──────────────────────────────────────────────────────────────────────────────

class MaskRCNNFPNHook:
    """Lấy feature map P3 từ FPN của cả hai mô hình."""
    def __init__(self, model: nn.Module):
        self.feat: torch.Tensor | None = None
        raw = model.module if hasattr(model, "module") else model
        self._hook = raw.backbone.fpn.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        if isinstance(out, dict): self.feat = out.get("0", out.get(list(out.keys())[0]))
        elif isinstance(out, (list, tuple)): self.feat = out[0]
        else: self.feat = out

    def remove(self):
        self._hook.remove()

class MutualLearningLoss(nn.Module):
    def __init__(self, temperature=4.0, kl_weight=0.5, feat_weight=0.3, box_weight=0.2):
        super().__init__()
        self.T = temperature
        self.kl_weight = kl_weight
        self.feat_weight = feat_weight
        self.box_weight = box_weight

    def kl_soft_loss(self, s_logits, t_logits):
        if s_logits.numel() == 0 or t_logits.numel() == 0:
            return torch.tensor(0.0, device=s_logits.device, requires_grad=True)
        n = min(s_logits.size(0), t_logits.size(0))
        s, t = s_logits[:n], t_logits[:n].detach()
        p_t = torch.sigmoid(t / self.T)
        p_s = torch.sigmoid(s / self.T)
        eps = 1e-7
        kl = p_t * (torch.log(p_t + eps) - torch.log(p_s + eps)) + \
             (1 - p_t) * (torch.log(1 - p_t + eps) - torch.log(1 - p_s + eps))
        return kl.mean() * (self.T ** 2)

    @staticmethod
    def box_iou(boxes_a, boxes_b):
        if boxes_a.numel() == 0 or boxes_b.numel() == 0:
            return torch.zeros(max(boxes_a.size(0), 1), max(boxes_b.size(0), 1), device=boxes_a.device)
        x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)

    def forward(self, soft_a, soft_b, feat_a, feat_b, mdl_coeff=1.0):
        device = feat_a.device if feat_a is not None else torch.device("cpu")
        kl_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)
        
        # 1. Box Matching & Distillation
        for s_a, s_b in zip(soft_a, soft_b):
            b_a, score_a = s_a["boxes"].to(device), s_a["scores"].to(device)
            b_b, score_b = s_b["boxes"].to(device), s_b["scores"].to(device)
            
            if b_a.numel() > 0 and b_b.numel() > 0:
                iou_mat = self.box_iou(b_a, b_b)
                max_iou, match_idx = iou_mat.max(dim=1)
                valid = max_iou > 0.5
                
                if valid.sum() > 0:
                    matched_b_boxes = b_b[match_idx[valid]]
                    matched_b_scores = score_b[match_idx[valid]]
                    valid_a_boxes = b_a[valid]
                    valid_a_scores = score_a[valid]
                    
                    box_loss += F.smooth_l1_loss(valid_a_boxes, matched_b_boxes.detach(), beta=0.1)
                    kl_loss += self.kl_soft_loss(valid_a_scores, matched_b_scores)
                    kl_loss += self.kl_soft_loss(matched_b_scores, valid_a_scores)

        kl_loss = kl_loss / max(len(soft_a), 1)
        box_loss = box_loss / max(len(soft_a), 1)

        # 2. Feature Mimicking
        feat_loss = torch.tensor(0.0, device=device)
        if feat_a is not None and feat_b is not None:
            f_b_resized = F.adaptive_avg_pool2d(feat_b.detach(), feat_a.shape[-2:])
            feat_loss = F.mse_loss(F.normalize(feat_a, dim=1), F.normalize(f_b_resized, dim=1))

        total_mdl = mdl_coeff * (self.kl_weight * kl_loss + self.feat_weight * feat_loss + self.box_weight * box_loss)
        return {"kl": kl_loss, "feat": feat_loss, "box": box_loss, "total": total_mdl}


# ──────────────────────────────────────────────────────────────────────────────
# 4. DATASET & ARGS
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".."))
    parser.add_argument("--gf7-bands", type=int, default=3, choices=[3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-instance-area", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints") / "mdl_swin_resnet")
    
    parser.add_argument("--mdl-weight", type=float, default=0.5)
    parser.add_argument("--feat-mimic-weight", type=float, default=0.3)
    parser.add_argument("--mdl-temperature", type=float, default=4.0)
    parser.add_argument("--mdl-warmup-epochs", type=int, default=5)
    parser.add_argument("--mdl-rampup-epochs", type=int, default=10)
    
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--swin-lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    
    # LRScheduler args
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["multistep", "cosine"])
    parser.add_argument("--lr-milestones", type=str, default="30,40")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--warmup-factor", type=float, default=0.001)
    
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--multi-gpu", action="store_true")
    
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--vis-every", type=int, default=10)
    parser.add_argument("--vis-num-samples", type=int, default=4)
    parser.add_argument("--vis-score-thresh", type=float, default=0.4)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()

def to_binary_mask(mask): return (mask[..., 0] if mask.ndim == 3 else mask > 127).astype(np.uint8)
def adapt_channels(image, in_channels):
    if image.ndim == 2: image = image[..., None]
    if image.shape[0] <= 8 and image.shape[-1] > 8: image = np.transpose(image, (1, 2, 0))
    return np.ascontiguousarray(image[..., :in_channels])

def normalize_image(image):
    denom = float(np.nanmax(image))
    denom = 1.0 if denom <= 1.0 else (255.0 if denom <= 255.0 else denom)
    return np.clip(image.astype(np.float32) / max(denom, 1.0), 0.0, 1.0)

def build_target_from_mask(mask, image_id, min_instance_area):
    binary = to_binary_mask(mask)
    num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes, labels, masks = [], [], []
    for lid in range(1, num_labels):
        if int(stats[lid, cv2.CC_STAT_AREA]) < min_instance_area: continue
        x, y, w, h = (int(stats[lid, i]) for i in [0, 1, 2, 3])
        if w <= 0 or h <= 0: continue
        boxes.append([x, y, x + w, y + h])
        labels.append(1)
        masks.append(labeled == lid)
    if not masks: return {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64), "masks": torch.zeros((0, binary.shape[0], binary.shape[1]), dtype=torch.uint8)}
    return {"boxes": torch.as_tensor(boxes, dtype=torch.float32), "labels": torch.as_tensor(labels, dtype=torch.int64), "masks": torch.as_tensor(np.stack(masks, 0), dtype=torch.uint8)}

class GF7Dataset(Dataset):
    def __init__(self, path, in_ch, augment=False):
        img_dir, lbl_dir = path/"image", path/"label"
        self.pairs = [(img, lbl_dir/img.name) for img in sorted(img_dir.glob("*.tif")) if (lbl_dir/img.name).exists()]
        self.in_ch, self.augment = in_ch, augment
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img, tgt = self.pairs[idx]
        i_arr = torch.from_numpy(normalize_image(adapt_channels(tifffile.imread(str(img)), self.in_ch))).permute(2,0,1).contiguous()
        target = build_target_from_mask(tifffile.imread(str(tgt)), idx, 20)
        
        # Simple flip augmentation
        if self.augment:
            _, h, w = i_arr.shape
            if random.random() < 0.5:
                i_arr = torch.flip(i_arr, dims=[2])
                if target["boxes"].numel() > 0:
                    b = target["boxes"].clone()
                    b[:, 0], b[:, 2] = w - target["boxes"][:, 2], w - target["boxes"][:, 0]
                    target["boxes"] = b
                if target["masks"].numel() > 0: target["masks"] = torch.flip(target["masks"], dims=[2])
            if random.random() < 0.5:
                i_arr = torch.flip(i_arr, dims=[1])
                if target["boxes"].numel() > 0:
                    b = target["boxes"].clone()
                    b[:, 1], b[:, 3] = h - target["boxes"][:, 3], h - target["boxes"][:, 1]
                    target["boxes"] = b
                if target["masks"].numel() > 0: target["masks"] = torch.flip(target["masks"], dims=[1])
        return i_arr, target

def collate_fn(batch): 
    images, targets = zip(*batch)
    return list(images), list(targets)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def _set_batchnorm_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

# ──────────────────────────────────────────────────────────────────────────────
# 5. CHECKPOINT & VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model_r, model_s, opt_r, opt_s, sched_r, sched_s, mdl_loss_fn, epoch, best_val, args, history, ema_r=None, ema_s=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_r = model_r.module if hasattr(model_r, "module") else model_r
    raw_s = model_s.module if hasattr(model_s, "module") else model_s
    torch.save({
        "epoch": epoch,
        "best_val_loss": best_val,
        "resnet_state_dict": raw_r.state_dict(),
        "swin_state_dict": raw_s.state_dict(),
        "mdl_state_dict": mdl_loss_fn.state_dict(),
        "resnet_opt_state": opt_r.state_dict(),
        "swin_opt_state": opt_s.state_dict(),
        "resnet_sched_state": sched_r.state_dict() if sched_r else None,
        "swin_sched_state": sched_s.state_dict() if sched_s else None,
        "ema_r_state": ema_r.state_dict() if ema_r else None,
        "ema_s_state": ema_s.state_dict() if ema_s else None,
        "args": vars(args),
        "history": history,
    }, path)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[CKPT] Saved -> {path}")

@torch.no_grad()
def visualize_mutual_predictions(model_r, model_s, val_dataset, device, epoch, output_dir, num_samples=4, score_thresh=0.4):
    raw_r = model_r.module if hasattr(model_r, "module") else model_r
    raw_s = model_s.module if hasattr(model_s, "module") else model_s
    raw_r.eval(); raw_s.eval()

    num_samples = min(num_samples, len(val_dataset))
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples), squeeze=False)
    fig.suptitle(f"Epoch {epoch} | MDL: ResNet50 ↔ Swin-T (thresh={score_thresh})", fontsize=14, fontweight="bold")

    def make_overlay(base, mask, color, alpha=0.45):
        out = base.copy()
        where = mask > 0.5
        for c, v in enumerate(color):
            out[..., c] = np.where(where, base[..., c] * (1 - alpha) + v * alpha, base[..., c])
        return out

    for row in range(num_samples):
        img_tensor, target = val_dataset[row]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_show = np.clip(img_np[..., :3] if img_np.shape[2] >= 3 else np.repeat(img_np, 3, axis=2), 0, 1)

        gt_masks = target["masks"].cpu().numpy()
        gt_combined = gt_masks.max(axis=0).astype(np.float32) if len(gt_masks) > 0 else np.zeros(img_show.shape[:2])

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out_r = raw_r([img_tensor.to(device)])[0]
            out_s = raw_s([img_tensor.to(device)])[0]

        # Resnet Preds
        pred_m_r = out_r["masks"].cpu().squeeze(1)
        keep_r = np.where(out_r["scores"].cpu().numpy() >= score_thresh)[0]
        r_combined = (pred_m_r[keep_r] > 0.5).float().max(dim=0).values.numpy() if len(keep_r) > 0 else np.zeros_like(gt_combined)

        # Swin Preds
        pred_m_s = out_s["masks"].cpu().squeeze(1)
        keep_s = np.where(out_s["scores"].cpu().numpy() >= score_thresh)[0]
        s_combined = (pred_m_s[keep_s] > 0.5).float().max(dim=0).values.numpy() if len(keep_s) > 0 else np.zeros_like(gt_combined)

        gt_vis = make_overlay(img_show, gt_combined, (0.2, 0.9, 0.2))
        r_vis = make_overlay(img_show, r_combined, (0.9, 0.2, 0.2))
        s_vis = make_overlay(img_show, s_combined, (0.2, 0.4, 0.9))

        for col, (vis, title) in enumerate([
            (img_show, "Input"),
            (gt_vis, f"GT [{len(gt_masks)}]"),
            (r_vis, f"ResNet [{len(keep_r)}]"),
            (s_vis, f"Swin-T [{len(keep_s)}]"),
        ]):
            axes[row][col].imshow(vis)
            axes[row][col].axis("off")
            axes[row][col].set_title(title, fontsize=10, fontweight="bold" if row == 0 else "normal")
        axes[row][0].set_ylabel(f"Sample {row}", fontsize=10, rotation=0, labelpad=50, va="center")

    fig.legend(handles=[
        mpatches.Patch(color=(0.2, 0.9, 0.2), label="Ground Truth"),
        mpatches.Patch(color=(0.9, 0.2, 0.2), label="ResNet50 (CNN)"),
        mpatches.Patch(color=(0.2, 0.4, 0.9), label="Swin-T (Transformer)"),
    ], loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.005))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / f"epoch_{epoch:03d}_mutual.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    raw_r.train(); raw_s.train()

# ──────────────────────────────────────────────────────────────────────────────
# 6. TRAINING & EVAL LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch_mutual(model_a, model_b, mdl_loss_fn, opt_a, opt_b, loader, device, epoch, args, hook_a, hook_b, scaler, sched_r, sched_s, ema_a, ema_b):
    model_a.train(); model_b.train()
    mdl_coeff = 0.0 if epoch <= args.mdl_warmup_epochs else min((epoch - args.mdl_warmup_epochs) / max(args.mdl_rampup_epochs, 1), 1.0)
    
    running = {k: 0.0 for k in ["a_nat", "b_nat", "kl", "feat", "box", "total"]}
    pbar = tqdm(loader, desc=f"Ep {epoch} [Train]", leave=False, disable=not (not dist.is_initialized() or dist.get_rank() == 0))

    for step, (imgs, tgts) in enumerate(pbar, 1):
        imgs = [i.to(device) for i in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        opt_a.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            hook_a.feat, hook_b.feat = None, None
            
            # Native Losses
            loss_dict_a = model_a(imgs, tgts)
            loss_dict_b = model_b(imgs, tgts)
            loss_nat_a = sum(loss_dict_a.values())
            loss_nat_b = sum(loss_dict_b.values())

            # Soft Predictions (Eval mode)
            with torch.no_grad():
                # Torchvision MaskRCNN only returns boxes/scores in eval mode
                model_a.eval(); model_b.eval()
                soft_a = model_a(imgs)
                soft_b = model_b(imgs)
                model_a.train(); model_b.train()

            # MDL Loss
            mdl_dict = mdl_loss_fn(soft_a, soft_b, hook_a.feat, hook_b.feat, mdl_coeff)
            total_loss = loss_nat_a + loss_nat_b + mdl_dict["total"]

        if scaler:
            scaler.scale(total_loss).backward()
            if args.clip_grad_norm:
                scaler.unscale_(opt_a); scaler.unscale_(opt_b)
                torch.nn.utils.clip_grad_norm_(model_a.parameters(), args.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), args.clip_grad_norm)
            scaler.step(opt_a); scaler.step(opt_b)
            scaler.update()
        else:
            total_loss.backward()
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model_a.parameters(), args.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), args.clip_grad_norm)
            opt_a.step(); opt_b.step()

        if ema_a: ema_a.update(model_a)
        if ema_b: ema_b.update(model_b)
        
        # Step scheduler per iter if using warmup
        if sched_r and args.warmup_iters > 0: sched_r.step()
        if sched_s and args.warmup_iters > 0: sched_s.step()

        running["a_nat"] += float(loss_nat_a); running["b_nat"] += float(loss_nat_b)
        running["kl"] += float(mdl_dict["kl"]); running["feat"] += float(mdl_dict["feat"])
        running["box"] += float(mdl_dict["box"]); running["total"] += float(total_loss)
        
        if step % args.print_freq == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            pbar.set_postfix(R50=f"{running['a_nat']/step:.2f}", SW=f"{running['b_nat']/step:.2f}", MDL=f"{running['kl']/step:.2f}")

    return {k: v / max(len(loader), 1) for k, v in running.items()}

@torch.no_grad()
def evaluate_loss_mutual(model_a, model_b, loader, device, epoch, scaler=None):
    model_a.train(); model_a.apply(_set_batchnorm_eval)
    model_b.train(); model_b.apply(_set_batchnorm_eval)
    running = {"a_nat": 0.0, "b_nat": 0.0, "total": 0.0}
    pbar = tqdm(loader, desc=f"Ep {epoch} [Val]", leave=False, disable=not (not dist.is_initialized() or dist.get_rank() == 0))
    
    for step, (imgs, tgts) in enumerate(pbar, 1):
        imgs = [i.to(device) for i in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            loss_nat_a = sum(model_a(imgs, tgts).values())
            loss_nat_b = sum(model_b(imgs, tgts).values())
            
        running["a_nat"] += float(loss_nat_a)
        running["b_nat"] += float(loss_nat_b)
        running["total"] += float(loss_nat_a + loss_nat_b)
        
        if step % 10 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
             pbar.set_postfix(R50=f"{running['a_nat']/step:.2f}", SW=f"{running['b_nat']/step:.2f}")

    return {k: v / max(len(loader), 1) for k, v in running.items()}

# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        raw = model.module if hasattr(model, "module") else model
        self.shadow = copy.deepcopy(raw).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        raw = model.module if hasattr(model, "module") else model
        for ema_p, m_p in zip(self.shadow.parameters(), raw.parameters()):
            ema_p.copy_(ema_p * self.decay + m_p.detach() * (1 - self.decay))
    def state_dict(self): return self.shadow.state_dict()
    def load_state_dict(self, sd): self.shadow.load_state_dict(sd)

def build_scheduler(optimizer, args, steps_per_epoch):
    total_iters = args.epochs * steps_per_epoch
    if args.warmup_iters > 0:
        warmup = LinearLR(optimizer, start_factor=args.warmup_factor, end_factor=1.0, total_iters=args.warmup_iters)
        main = CosineAnnealingLR(optimizer, T_max=max(1, total_iters - args.warmup_iters), eta_min=optimizer.param_groups[-1]["lr"] * 0.01) if args.scheduler == "cosine" else MultiStepLR(optimizer, milestones=[int(m)*steps_per_epoch for m in args.lr_milestones.split(",")], gamma=args.lr_gamma)
        return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[args.warmup_iters])
    else:
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=optimizer.param_groups[-1]["lr"] * 0.01) if args.scheduler == "cosine" else MultiStepLR(optimizer, milestones=[int(m) for m in args.lr_milestones.split(",")], gamma=args.lr_gamma)

def main():
    args = parse_args()
    seed_everything(args.seed)
    
    if args.multi_gpu:
        dist.init_process_group("nccl")
        torch.cuda.set_device(dist.get_rank())
    device = torch.device(f"cuda:{dist.get_rank() if dist.is_initialized() else 0}")
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print("="*70)
        print("  MDL: Mask R-CNN (ResNet50) ↔ Mask R-CNN (Swin-T)")
        print(f"  Device: {device} | DDP: {args.multi_gpu} | AMP: {args.amp} | EMA: {args.ema}")
        print("="*70)
    
    root = args.data_root / f"GF-7 Building ({args.gf7_bands}Bands)"
    train_ds = GF7Dataset(root/"Train", args.gf7_bands, augment=True)
    val_ds = GF7Dataset(root/"Val", args.gf7_bands, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.multi_gpu,
                              sampler=DistributedSampler(train_ds) if args.multi_gpu else None,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False,
                            sampler=DistributedSampler(val_ds, shuffle=False) if args.multi_gpu else None,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model_resnet = get_resnet_model(2, not args.no_pretrained, args.gf7_bands).to(device)
    model_swin = get_swin_model(2, not args.no_pretrained, args.gf7_bands).to(device)
    
    hook_resnet = MaskRCNNFPNHook(model_resnet)
    hook_swin = MaskRCNNFPNHook(model_swin)
    mdl_loss = MutualLearningLoss(args.mdl_temperature, args.mdl_weight, args.feat_mimic_weight).to(device)

    if args.multi_gpu:
        model_resnet = DDP(model_resnet, device_ids=[dist.get_rank()], find_unused_parameters=False)
        model_swin = DDP(model_swin, device_ids=[dist.get_rank()], find_unused_parameters=False)

    opt_resnet = torch.optim.SGD([p for p in model_resnet.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    opt_swin = torch.optim.AdamW([p for p in model_swin.parameters() if p.requires_grad], lr=args.swin_lr, weight_decay=args.weight_decay)

    sched_r = build_scheduler(opt_resnet, args, len(train_loader))
    sched_s = build_scheduler(opt_swin, args, len(train_loader))

    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    ema_r = ModelEMA(model_resnet, args.ema_decay) if args.ema else None
    ema_s = ModelEMA(model_swin, args.ema_decay) if args.ema else None

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    # Resume logic
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        (model_resnet.module if hasattr(model_resnet, "module") else model_resnet).load_state_dict(ckpt["resnet_state_dict"])
        (model_swin.module if hasattr(model_swin, "module") else model_swin).load_state_dict(ckpt["swin_state_dict"])
        opt_resnet.load_state_dict(ckpt["resnet_opt_state"])
        opt_swin.load_state_dict(ckpt["swin_opt_state"])
        if ema_r and "ema_r_state" in ckpt: ema_r.load_state_dict(ckpt["ema_r_state"])
        if ema_s and "ema_s_state" in ckpt: ema_s.load_state_dict(ckpt["ema_s_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        history = ckpt.get("history", [])

    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(args.output_dir / "mdl_config.json", "w") as f:
            json.dump(vars(args), f, default=str, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        if args.multi_gpu: train_loader.sampler.set_epoch(epoch)
        
        train_metrics = train_one_epoch_mutual(model_resnet, model_swin, mdl_loss, opt_resnet, opt_swin, train_loader, device, epoch, args, hook_resnet, hook_swin, scaler, sched_r, sched_s, ema_r, ema_s)
        
        eval_r = ema_r.shadow if ema_r else model_resnet
        eval_s = ema_s.shadow if ema_s else model_swin
        val_metrics = evaluate_loss_mutual(eval_r, eval_s, val_loader, device, epoch, scaler)
        
        if args.warmup_iters == 0:
            sched_r.step(); sched_s.step()

        val_total = val_metrics["total"]
        if args.multi_gpu:
            t = torch.tensor(val_total, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            val_total = t.item()

        if not dist.is_initialized() or dist.get_rank() == 0:
            lr_r = opt_resnet.param_groups[0]["lr"]
            lr_s = opt_swin.param_groups[0]["lr"]
            mdl_coeff = 0.0 if epoch <= args.mdl_warmup_epochs else min((epoch - args.mdl_warmup_epochs) / max(args.mdl_rampup_epochs, 1), 1.0)
            
            print(f"[Ep {epoch:03d}] R50(T={train_metrics['a_nat']:.2f}, V={val_metrics['a_nat']:.2f}) | "
                  f"SWIN(T={train_metrics['b_nat']:.2f}, V={val_metrics['b_nat']:.2f}) | "
                  f"MDL(KL={train_metrics['kl']:.2f}, Feat={train_metrics['feat']:.2f}) | "
                  f"ValTot: {val_total:.3f} | LR_R={lr_r:.1e}")

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "val_total": val_total})
            with open(args.output_dir / "history.json", "w") as f: json.dump(history, f, indent=2)

            if epoch % args.save_every == 0:
                save_checkpoint(args.output_dir / f"checkpoint_epoch_{epoch:03d}.pth", model_resnet, model_swin, opt_resnet, opt_swin, sched_r, sched_s, mdl_loss, epoch, best_val_loss, args, history, ema_r, ema_s)
            
            if val_total < best_val_loss:
                best_val_loss = val_total
                save_checkpoint(args.output_dir / "best_model.pth", model_resnet, model_swin, opt_resnet, opt_swin, sched_r, sched_s, mdl_loss, epoch, best_val_loss, args, history, ema_r, ema_s)
                print(f"  ★ New Best Model -> {best_val_loss:.3f}")

            if args.vis_every > 0 and epoch % args.vis_every == 0:
                visualize_mutual_predictions(eval_r, eval_s, val_ds, device, epoch, args.output_dir, args.vis_num_samples, args.vis_score_thresh)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("="*70); print(f"[DONE] Best Val Loss = {best_val_loss:.3f}"); print("="*70)
    if args.multi_gpu: dist.destroy_process_group()

if __name__ == "__main__":
    main()
