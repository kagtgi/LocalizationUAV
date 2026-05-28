import argparse
import copy
import json
import os
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, an toàn khi dùng DDP
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import get_model

# ──────────────────────────────────────────────────────────────────────────────
# Ví dụ chạy (ưu tiên 3Bands):
#
#   Single-GPU:
#     python3 train_maskrcnn.py --data-root .. --gf7-bands 3 --epochs 50 --batch-size 4
#
#   Multi-GPU (khuyến nghị, sau khi cài PyTorch nightly cu128):
#     torchrun --nproc_per_node=2 train_maskrcnn.py \
#       --data-root .. --gf7-bands 3 --epochs 50 --batch-size 4 \
#       --multi-gpu --amp --scale-lr \
#       --layerwise-lr --warmup-iters 500 --scheduler cosine \
#       --ema --clip-grad-norm 1.0 \
#       --vis-every 10 --vis-num-samples 4
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    default_data_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN on GF-7 building datasets — toi uu cho 3Bands."
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--gf7-bands", type=int, default=3, choices=[3, 4],
                        help="Uu tien 3Bands (mac dinh=3). 4Bands neu can kenh NIR.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size PER GPU. Effective = batch-size x num_gpus.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-instance-area", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("checkpoints") / "maskrcnn_gf7_3bands")

    # ── Multi-GPU ──────────────────────────────────────────────────────────────
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Bat DistributedDataParallel (DDP).")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Chi dinh GPU, vd: '0,1'. Chi dung khi chay bang python (khong torchrun).")
    parser.add_argument("--scale-lr", action="store_true",
                        help="Nhan LR tuyen tinh theo so GPU (linear scaling rule).")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"],
                        help="SGD phu hop hon cho Mask R-CNN pretrained. AdamW cho fine-tune nhanh.")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Base LR cho head/FPN/RPN. Backbone se nho hon neu --layerwise-lr.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    # ── Layer-wise LR ─────────────────────────────────────────────────────────
    parser.add_argument("--layerwise-lr", action="store_true",
                        help="Chia LR theo layer: backbone.body x backbone-lr-scale, "
                             "FPN/RPN/heads x LR day du.")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1,
                        help="He so nhan LR cho backbone.body (mac dinh 0.1 = 1/10 LR).")

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["multistep", "cosine"],
                        help="cosine (mac dinh) hoac multistep.")
    parser.add_argument("--lr-milestones", type=str, default="30,40",
                        help="Dung voi --scheduler multistep.")
    parser.add_argument("--lr-gamma", type=float, default=0.1)

    # ── Warmup ────────────────────────────────────────────────────────────────
    parser.add_argument("--warmup-iters", type=int, default=500,
                        help="So iteration warmup (LinearLR tu LR/1000 -> LR). 0 = tat.")
    parser.add_argument("--warmup-factor", type=float, default=0.001,
                        help="start_factor cua LinearLR warmup.")

    # ── AMP ───────────────────────────────────────────────────────────────────
    parser.add_argument("--amp", action="store_true",
                        help="Bat Automatic Mixed Precision (FP16). Khuyen nghi cho RTX GPU.")

    # ── Gradient clipping ─────────────────────────────────────────────────────
    parser.add_argument("--clip-grad-norm", type=float, default=1.0,
                        help="Max norm gradient clipping. 0 = tat.")

    # ── EMA ───────────────────────────────────────────────────────────────────
    parser.add_argument("--ema", action="store_true",
                        help="Bat Exponential Moving Average cua model weights. "
                             "Cai thien val loss va generalization.")
    parser.add_argument("--ema-decay", type=float, default=0.9998,
                        help="EMA decay factor (0.9998 phu hop batch nho).")

    # ── Extra data ────────────────────────────────────────────────────────────
    parser.add_argument("--include-images-shp", action="store_true")
    parser.add_argument("--images-shp-root", type=Path, default=None)
    parser.add_argument("--extra-patch-size", type=int, default=512)
    parser.add_argument("--extra-patch-stride", type=int, default=384)
    parser.add_argument("--extra-min-fg-ratio", type=float, default=0.002)
    parser.add_argument("--extra-bg-keep-prob", type=float, default=0.03)
    parser.add_argument("--extra-max-patches-per-area", type=int, default=1500)
    parser.add_argument("--extra-val-ratio", type=float, default=0.0)

    # ── Visualization ──────────────────────────────────────────────────────────
    parser.add_argument("--vis-every", type=int, default=10,
                        help="Luu anh pred vs GT moi N epochs. 0 = tat.")
    parser.add_argument("--vis-num-samples", type=int, default=4,
                        help="So anh val de visualize.")
    parser.add_argument("--vis-score-thresh", type=float, default=0.4,
                        help="Score threshold khi ve predicted masks.")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_ddp(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Giu ban shadow-copy cua model weights duoc average theo thoi gian.
    Dung EMA model de evaluate -> val loss on dinh hon, it dao dong.
    Khi resume, EMA state cung duoc luu/load cung checkpoint.

    Update rule: shadow = decay * shadow + (1 - decay) * model
    """
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        raw = model.module if hasattr(model, "module") else model
        self.shadow = copy.deepcopy(raw)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        raw = model.module if hasattr(model, "module") else model
        for ema_p, m_p in zip(self.shadow.parameters(), raw.parameters()):
            ema_p.copy_(ema_p * self.decay + m_p.detach() * (1.0 - self.decay))
        for ema_b, m_b in zip(self.shadow.buffers(), raw.buffers()):
            ema_b.copy_(m_b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer: Layer-wise LR
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, args, effective_lr):
    """
    Layer-wise LR strategy cho ResNet50-FPN Mask R-CNN:
      - backbone.body  (ResNet-50): lr * backbone_lr_scale  <- tinh chinh nhe
      - backbone.fpn, rpn, roi_heads: lr day du             <- hoc manh hon
    """
    raw_model = model.module if hasattr(model, "module") else model

    if args.layerwise_lr:
        backbone_params, other_params = [], []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone.body" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        backbone_lr = effective_lr * args.backbone_lr_scale
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr,  "name": "backbone"},
            {"params": other_params,    "lr": effective_lr, "name": "heads"},
        ]
        if is_main_process():
            n_bb = sum(p.numel() for p in backbone_params)
            n_hd = sum(p.numel() for p in other_params)
            print(f"[INFO] Layer-wise LR:")
            print(f"       backbone.body  : {n_bb:,} params @ lr={backbone_lr:.6f}")
            print(f"       FPN/RPN/heads  : {n_hd:,} params @ lr={effective_lr:.6f}")
    else:
        param_groups = [p for p in raw_model.parameters() if p.requires_grad]

    if args.optimizer == "sgd":
        return torch.optim.SGD(
            param_groups, lr=effective_lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
        )
    else:
        return torch.optim.AdamW(
            param_groups, lr=effective_lr, weight_decay=args.weight_decay,
        )


# ──────────────────────────────────────────────────────────────────────────────
# LR Scheduler: Warmup + Cosine/MultiStep
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, args, steps_per_epoch):
    """
    Tra ve (scheduler, step_per_iter: bool).
    - step_per_iter=True  : goi scheduler.step() sau moi batch (warmup phase)
    - step_per_iter=False : goi sau moi epoch
    """
    total_iters = args.epochs * steps_per_epoch
    use_warmup  = args.warmup_iters > 0

    if use_warmup:
        warmup_sched = LinearLR(
            optimizer,
            start_factor=args.warmup_factor,
            end_factor=1.0,
            total_iters=args.warmup_iters,
        )
        remaining = max(1, total_iters - args.warmup_iters)
        base_lr   = optimizer.param_groups[-1]["lr"]  # dung group cuoi (heads)

        if args.scheduler == "cosine":
            main_sched = CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=base_lr * 0.01)
        else:
            mile_iters = [int(m.strip()) * steps_per_epoch
                          for m in args.lr_milestones.split(",") if m.strip()]
            main_sched = MultiStepLR(
                optimizer, milestones=mile_iters, gamma=args.lr_gamma)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[args.warmup_iters],
        )
        if is_main_process():
            print(f"[INFO] Scheduler: warmup({args.warmup_iters} iter, "
                  f"factor={args.warmup_factor}) -> {args.scheduler}({remaining} iter)")
        return scheduler, True

    else:
        base_lr = optimizer.param_groups[-1]["lr"]
        if args.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=base_lr * 0.01)
        else:
            milestones = [int(m.strip()) for m in args.lr_milestones.split(",") if m.strip()]
            scheduler  = MultiStepLR(
                optimizer, milestones=milestones, gamma=args.lr_gamma)
        if is_main_process():
            print(f"[INFO] Scheduler: {args.scheduler} (per-epoch, no warmup)")
        return scheduler, False


# ──────────────────────────────────────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


# ──────────────────────────────────────────────────────────────────────────────
# Data utilities (giu nguyen logic goc, khong doi duong dan / format)
# ──────────────────────────────────────────────────────────────────────────────

def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def resolve_gf7_root(data_root, bands):
    return data_root / ("GF-7 Building (3Bands)" if bands == 3 else "GF-7 Building (4Bands)")


def collect_pairs(split_root):
    image_dir = split_root / "image"
    label_dir = split_root / "label"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing split folders: {image_dir} or {label_dir}")
    image_paths = sorted(image_dir.glob("*.tif"))
    pairs, missing = [], 0
    for img_path in image_paths:
        lbl_path = label_dir / img_path.name
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            missing += 1
    if not pairs:
        raise RuntimeError(f"No image-label pairs found under {split_root}")
    if missing > 0:
        print(f"[WARN] {missing} images had no matching label under {split_root}.")
    return pairs


def to_channel_last(image):
    if image.ndim == 2:
        return image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Unsupported ndim={image.ndim}")
    if image.shape[0] <= 8 and image.shape[-1] > 8:
        image = np.transpose(image, (1, 2, 0))
    return image


def adapt_channels(image, in_channels):
    image = to_channel_last(image)
    c = image.shape[-1]
    if c == in_channels:
        return np.ascontiguousarray(image)
    if c > in_channels:
        return np.ascontiguousarray(image[..., :in_channels])
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
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 127).astype(np.uint8)


def build_target_from_mask(mask, image_id, min_instance_area):
    binary = to_binary_mask(mask)
    num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes, labels, masks, areas = [], [], [], []
    for lid in range(1, num_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_instance_area:
            continue
        x = int(stats[lid, cv2.CC_STAT_LEFT])
        y = int(stats[lid, cv2.CC_STAT_TOP])
        w = int(stats[lid, cv2.CC_STAT_WIDTH])
        h = int(stats[lid, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
        labels.append(1)
        masks.append(labeled == lid)
        areas.append(float(area))

    if masks:
        return {
            "boxes":    torch.as_tensor(boxes,                  dtype=torch.float32),
            "labels":   torch.as_tensor(labels,                 dtype=torch.int64),
            "masks":    torch.as_tensor(np.stack(masks, 0),     dtype=torch.uint8),
            "image_id": torch.tensor([image_id],                dtype=torch.int64),
            "area":     torch.as_tensor(areas,                  dtype=torch.float32),
            "iscrowd":  torch.zeros((len(masks),),              dtype=torch.int64),
        }
    else:
        h, w = binary.shape
        return {
            "boxes":    torch.zeros((0, 4),    dtype=torch.float32),
            "labels":   torch.zeros((0,),      dtype=torch.int64),
            "masks":    torch.zeros((0, h, w), dtype=torch.uint8),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area":     torch.zeros((0,),      dtype=torch.float32),
            "iscrowd":  torch.zeros((0,),      dtype=torch.int64),
        }


def apply_random_flips(image, target):
    _, h, w = image.shape
    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])
        if target["boxes"].numel() > 0:
            b = target["boxes"].clone()
            b[:, 0] = w - target["boxes"][:, 2]
            b[:, 2] = w - target["boxes"][:, 0]
            target["boxes"] = b
        if target["masks"].numel() > 0:
            target["masks"] = torch.flip(target["masks"], dims=[2])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[1])
        if target["boxes"].numel() > 0:
            b = target["boxes"].clone()
            b[:, 1] = h - target["boxes"][:, 3]
            b[:, 3] = h - target["boxes"][:, 1]
            target["boxes"] = b
        if target["masks"].numel() > 0:
            target["masks"] = torch.flip(target["masks"], dims=[1])
    return image, target


# ──────────────────────────────────────────────────────────────────────────────
# Dataset classes (giu nguyen cau truc duong dan goc)
# ──────────────────────────────────────────────────────────────────────────────

class GF7MaskRCNNDataset(Dataset):
    def __init__(self, pairs, in_channels, min_instance_area=20,
                 augment=False, image_id_offset=0):
        self.pairs             = pairs
        self.in_channels       = in_channels
        self.min_instance_area = min_instance_area
        self.augment           = augment
        self.image_id_offset   = image_id_offset

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, lbl_path = self.pairs[index]
        image  = tifffile.imread(str(img_path))
        mask   = tifffile.imread(str(lbl_path))
        image  = adapt_channels(image, self.in_channels)
        image  = normalize_image(image)
        image  = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        target = build_target_from_mask(
            mask, self.image_id_offset + index, self.min_instance_area)
        if self.augment:
            image, target = apply_random_flips(image, target)
        return image, target


def sliding_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def collect_area_patch_windows(area_root, patch_size, stride, min_fg_ratio,
                                bg_keep_prob, max_patches_per_area, seed):
    area_root = Path(area_root)
    area_dirs = sorted([p for p in area_root.glob("*/*") if p.is_dir()])
    if not area_dirs:
        print(f"[WARN] No city/area folders found under {area_root}.")
        return []
    rng = random.Random(seed)
    windows = []
    print(f"[INFO] Building extra patch index from {area_root} ...")
    for area_dir in area_dirs:
        img_path = area_dir / "image_no_georef.tif"
        lbl_path = area_dir / "label_no_georef.tif"
        if not (img_path.exists() and lbl_path.exists()):
            continue
        label = to_binary_mask(tifffile.imread(str(lbl_path)))
        h, w  = label.shape
        area_windows = []
        for top in sliding_positions(h, patch_size, stride):
            for left in sliding_positions(w, patch_size, stride):
                patch = label[top:top + patch_size, left:left + patch_size]
                if float((patch > 0).mean()) >= min_fg_ratio or rng.random() < bg_keep_prob:
                    area_windows.append((img_path, lbl_path, left, top, patch_size))
        if max_patches_per_area > 0 and len(area_windows) > max_patches_per_area:
            rng.shuffle(area_windows)
            area_windows = area_windows[:max_patches_per_area]
        windows.extend(area_windows)
        print(f"  - {area_dir.parent.name}/{area_dir.name}: {len(area_windows)} patches")
    rng.shuffle(windows)
    print(f"[INFO] Total extra patches: {len(windows)}")
    return windows


class AreaWindowMaskRCNNDataset(Dataset):
    def __init__(self, windows, in_channels, min_instance_area=20,
                 augment=False, image_id_offset=0):
        self.windows           = windows
        self.in_channels       = in_channels
        self.min_instance_area = min_instance_area
        self.augment           = augment
        self.image_id_offset   = image_id_offset
        self._img_readers      = {}
        self._lbl_readers      = {}

    def __len__(self):
        return len(self.windows)

    def _get_reader(self, path, cache):
        p = str(path)
        if p not in cache:
            import rasterio
            cache[p] = rasterio.open(p)
        return cache[p]

    @staticmethod
    def _pad_to_patch(arr, patch_size, fill=0):
        if arr.ndim == 2:
            h, w = arr.shape
            if h == patch_size and w == patch_size:
                return arr
            out = np.full((patch_size, patch_size), fill, dtype=arr.dtype)
            out[:h, :w] = arr
            return out
        h, w, c = arr.shape
        if h == patch_size and w == patch_size:
            return arr
        out = np.full((patch_size, patch_size, c), fill, dtype=arr.dtype)
        out[:h, :w] = arr
        return out

    def __getitem__(self, index):
        img_path, lbl_path, left, top, patch_size = self.windows[index]
        import rasterio.windows
        win   = rasterio.windows.Window(left, top, patch_size, patch_size)
        image = np.transpose(
            self._get_reader(img_path, self._img_readers).read(window=win), (1, 2, 0))
        mask  = self._get_reader(lbl_path, self._lbl_readers).read(1, window=win)
        image = self._pad_to_patch(image, patch_size)
        mask  = self._pad_to_patch(mask,  patch_size)
        image = adapt_channels(image, self.in_channels)
        image = normalize_image(image)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        target = build_target_from_mask(
            mask, self.image_id_offset + index, self.min_instance_area)
        if self.augment:
            image, target = apply_random_flips(image, target)
        return image, target

    def __del__(self):
        for cache in (self._img_readers, self._lbl_readers):
            for r in cache.values():
                try:
                    r.close()
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# Training / Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def move_batch_to_device(images, targets, device):
    images  = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
               for t in targets]
    return images, targets


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
                    scaler=None, clip_grad_norm=None,
                    scheduler=None, step_per_iter=False, ema=None):
    model.train()
    running_loss = 0.0
    num_batches  = len(data_loader)
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [train]",
                leave=False, disable=not is_main_process())

    for step, (images, targets) in enumerate(pbar, start=1):
        images, targets = move_batch_to_device(images, targets, device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())
            scaler.scale(losses).backward()
            if clip_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())
            losses.backward()
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        # EMA update sau moi step
        if ema is not None:
            ema.update(model)

        # Warmup scheduler: step theo iteration
        if step_per_iter and scheduler is not None:
            scheduler.step()

        running_loss += float(losses.item())
        if is_main_process() and (step % print_freq == 0 or step == num_batches):
            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{running_loss / step:.4f}", lr=f"{lr_now:.2e}")

    return running_loss / max(num_batches, 1)


def _set_batchnorm_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


@torch.no_grad()
def evaluate_loss(model, data_loader, device, epoch, scaler=None):
    """
    Mask R-CNN tra loss trong train mode + co targets.
    BatchNorm bi freeze ve eval de dam bao running stats on dinh.
    """
    model.train()
    model.apply(_set_batchnorm_eval)
    running_loss = 0.0
    num_batches  = len(data_loader)
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [val]",
                leave=False, disable=not is_main_process())

    for step, (images, targets) in enumerate(pbar, start=1):
        images, targets = move_batch_to_device(images, targets, device)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())
        running_loss += float(losses.item())
        if is_main_process():
            pbar.set_postfix(loss=f"{running_loss / step:.4f}")

    return running_loss / max(num_batches, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization: Predicted vs Ground Truth
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def visualize_predictions(model, val_dataset, device, epoch, output_dir,
                          num_samples=4, score_thresh=0.4):
    """
    So sanh Predicted masks vs Ground Truth cho num_samples anh val dau tien.
    Luu tai output_dir/vis/epoch_NNN.png — chi chay o rank-0.
    Dung EMA model neu duoc truyen vao (on dinh hon).
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    num_samples = min(num_samples, len(val_dataset))
    fig, axes   = plt.subplots(num_samples, 3,
                               figsize=(15, 5 * num_samples), squeeze=False)
    fig.suptitle(f"Epoch {epoch}  |  Predicted Masks vs Ground Truth  "
                 f"(score >= {score_thresh})",
                 fontsize=13, fontweight="bold")

    def make_overlay(base, mask, color, alpha=0.45):
        out   = base.copy()
        where = mask > 0.5
        for c, v in enumerate(color):
            out[..., c] = np.where(
                where, base[..., c] * (1 - alpha) + v * alpha, base[..., c])
        return out

    for row in range(num_samples):
        image_tensor, target = val_dataset[row]

        # Anh goc (dung 3 kenh dau de hien thi)
        img_np   = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_show = img_np[..., :3] if img_np.shape[2] >= 3 \
                   else np.repeat(img_np, 3, axis=2)
        img_show = np.clip(img_show, 0, 1)

        # Ground Truth
        gt_masks    = target["masks"].cpu().numpy()
        gt_combined = (gt_masks.max(axis=0).astype(np.float32)
                       if len(gt_masks) > 0
                       else np.zeros(img_show.shape[:2], dtype=np.float32))

        # Inference (dung EMA model neu duoc truyen vao)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = raw_model([image_tensor.to(device)])[0]

        pred_masks_all = out["masks"].cpu().squeeze(1)  # (N, H, W) sigmoid
        pred_scores    = out["scores"].cpu().numpy()
        keep_idx       = np.where(pred_scores >= score_thresh)[0]

        if len(keep_idx) > 0:
            pred_combined = (pred_masks_all[keep_idx] > 0.5).float() \
                                .max(dim=0).values.numpy()
        else:
            pred_combined = np.zeros(img_show.shape[:2], dtype=np.float32)

        gt_vis   = make_overlay(img_show, gt_combined,   color=(0.2, 0.9, 0.2))
        pred_vis = make_overlay(img_show, pred_combined, color=(0.9, 0.2, 0.2))

        n_gt   = len(gt_masks)
        n_pred = len(keep_idx)
        n_all  = len(pred_scores)

        for col, (vis, subtitle) in enumerate([
            (img_show, "Input (RGB / 3ch)"),
            (gt_vis,   f"Ground Truth  [{n_gt} bldg]"),
            (pred_vis, f"Predicted  [{n_pred}/{n_all} dets]"),
        ]):
            ax = axes[row][col]
            ax.imshow(vis)
            ax.axis("off")
            ax.set_title(subtitle, fontsize=9 + (row == 0),
                         fontweight="bold" if row == 0 else "normal")
        axes[row][0].set_ylabel(f"Sample {row}", fontsize=9,
                                rotation=0, labelpad=45, va="center")

    fig.legend(handles=[
        mpatches.Patch(color=(0.2, 0.9, 0.2), label="Ground Truth"),
        mpatches.Patch(color=(0.9, 0.2, 0.2),
                       label=f"Prediction (score >= {score_thresh})"),
    ], loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    vis_dir   = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_path = vis_dir / f"epoch_{epoch:03d}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] Saved -> {save_path}")
    raw_model.train()


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val_loss, args, history, ema=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch":                epoch,
        "best_val_loss":        best_val_loss,
        "model_state_dict":     raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "ema_state_dict":       ema.state_dict() if ema is not None else None,
        "args":                 vars(args),
        "history":              history,
    }, path)


# ──────────────────────────────────────────────────────────────────────────────
# Main train function
# ──────────────────────────────────────────────────────────────────────────────

def train(rank, world_size, args):
    use_ddp = args.multi_gpu and world_size > 1

    if use_ddp:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(
            "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)

    if is_main_process():
        print("=" * 65)
        print(f"  GF-7 Mask R-CNN Training  |  Bands: {args.gf7_bands}  "
              f"(uu tien 3Bands)")
        print(f"  Device : {device}  |  World : {world_size}  |  DDP : {use_ddp}")
        print(f"  AMP    : {args.amp}  |  EMA : {args.ema}  "
              f"|  LayerwiseLR : {args.layerwise_lr}")
        print(f"  Sched  : {args.scheduler}  "
              f"|  Warmup : {args.warmup_iters} iters")
        print("=" * 65)

    # ── Datasets ──────────────────────────────────────────────────────────────
    data_root  = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    gf7_root    = resolve_gf7_root(data_root, args.gf7_bands)
    train_pairs = collect_pairs(gf7_root / "Train")
    val_pairs   = collect_pairs(gf7_root / "Val")
    in_channels = 3 if args.gf7_bands == 3 else 4
    image_id_cursor = 0

    train_dataset = GF7MaskRCNNDataset(
        train_pairs, in_channels, args.min_instance_area,
        augment=True, image_id_offset=image_id_cursor)
    image_id_cursor += len(train_dataset)

    val_dataset = GF7MaskRCNNDataset(
        val_pairs, in_channels, args.min_instance_area,
        augment=False, image_id_offset=image_id_cursor)
    image_id_cursor += len(val_dataset)

    if args.include_images_shp:
        images_shp_root = args.images_shp_root or (data_root / "Images and Shpfiles")
        extra_windows = collect_area_patch_windows(
            images_shp_root, args.extra_patch_size, args.extra_patch_stride,
            args.extra_min_fg_ratio, args.extra_bg_keep_prob,
            args.extra_max_patches_per_area, args.seed)
        if extra_windows:
            split_at = int(len(extra_windows) * (1.0 - args.extra_val_ratio))
            extra_train_ds = AreaWindowMaskRCNNDataset(
                extra_windows[:split_at], in_channels, args.min_instance_area,
                augment=True, image_id_offset=image_id_cursor)
            image_id_cursor += len(extra_train_ds)
            train_dataset = ConcatDataset([train_dataset, extra_train_ds])
            if args.extra_val_ratio > 0:
                extra_val_ds = AreaWindowMaskRCNNDataset(
                    extra_windows[split_at:], in_channels, args.min_instance_area,
                    augment=False, image_id_offset=image_id_cursor)
                image_id_cursor += len(extra_val_ds)
                val_dataset = ConcatDataset([val_dataset, extra_val_ds])

    if is_main_process():
        print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)} "
              f"| Channels: {in_channels}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_sampler = (DistributedSampler(train_dataset, num_replicas=world_size,
                                        rank=rank, shuffle=True, seed=args.seed)
                     if use_ddp else None)
    val_sampler   = (DistributedSampler(val_dataset, num_replicas=world_size,
                                        rank=rank, shuffle=False)
                     if use_ddp else None)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=max(1, args.batch_size // 2),
        sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(num_classes=2, pretrained=not args.no_pretrained,
                      in_channels=in_channels)
    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Optimizer (layer-wise LR) ──────────────────────────────────────────────
    effective_lr = args.lr * world_size if args.scale_lr else args.lr
    if is_main_process() and args.scale_lr:
        print(f"[INFO] LR scaled: {args.lr} x {world_size} = {effective_lr:.6f}")

    optimizer = build_optimizer(model, args, effective_lr)

    # ── LR Scheduler (warmup + cosine/multistep) ───────────────────────────────
    steps_per_epoch           = len(train_loader)
    scheduler, step_per_iter  = build_scheduler(optimizer, args, steps_per_epoch)

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    scaler = (torch.amp.GradScaler("cuda")
              if args.amp and device.type == "cuda" else None)

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    history       = []

    if args.resume is not None:
        ckpt = torch.load(args.resume.resolve(), map_location=device)
        raw  = model.module if hasattr(model, "module") else model
        raw.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ema is not None and ckpt.get("ema_state_dict"):
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch   = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        history       = list(ckpt.get("history", []))
        if is_main_process():
            print(f"[INFO] Resumed from {args.resume} @ epoch {start_epoch}")

    if is_main_process():
        with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model=model, optimizer=optimizer, data_loader=train_loader,
            device=device, epoch=epoch, print_freq=args.print_freq,
            scaler=scaler, clip_grad_norm=args.clip_grad_norm,
            scheduler=scheduler, step_per_iter=step_per_iter, ema=ema,
        )

        # Evaluate voi EMA model (neu co) de val loss on dinh hon
        eval_model = ema.shadow if ema is not None else model
        val_loss   = evaluate_loss(eval_model, val_loader, device, epoch, scaler)

        # Scheduler step theo epoch (neu khong phai per-iter)
        if not step_per_iter:
            scheduler.step()

        # Dong bo val_loss giua cac GPU
        if use_ddp:
            t = torch.tensor(val_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            val_loss = t.item()

        lr_now = optimizer.param_groups[0]["lr"]

        if is_main_process():
            ema_tag = " [EMA val]" if ema is not None else ""
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val_loss, "lr": lr_now})
            print(f"[EPOCH {epoch:03d}/{args.epochs}] "
                  f"train={train_loss:.4f}  val={val_loss:.4f}{ema_tag}  "
                  f"lr={lr_now:.2e}")

            with open(output_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            if epoch % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_epoch_{epoch:03d}.pth",
                    model, optimizer, scheduler, epoch,
                    best_val_loss, args, history, ema)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    output_dir / "best_model.pth",
                    model, optimizer, scheduler, epoch,
                    best_val_loss, args, history, ema)
                print(f"[INFO] Best model -> val_loss={best_val_loss:.4f}")

            # Visualization: dung EMA model neu co
            if args.vis_every > 0 and epoch % args.vis_every == 0:
                vis_model = ema.shadow if ema is not None else model
                visualize_predictions(
                    vis_model, val_dataset, device, epoch, output_dir,
                    num_samples=args.vis_num_samples,
                    score_thresh=args.vis_score_thresh,
                )

    if is_main_process():
        print("=" * 65)
        print(f"[DONE] Best val_loss = {best_val_loss:.4f}")
        print(f"[DONE] Outputs: {output_dir}")
        print("=" * 65)

    if use_ddp:
        cleanup_ddp()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not (0.0 <= args.extra_val_ratio <= 0.5):
        raise ValueError("--extra-val-ratio phai nam trong [0.0, 0.5].")

    seed_everything(args.seed)

    if args.multi_gpu:
        if args.gpu_ids is not None:
            gpu_list = [int(g.strip()) for g in args.gpu_ids.split(",")]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
            world_size = len(gpu_list)
        else:
            world_size = torch.cuda.device_count()

        if world_size < 2:
            print("[WARN] --multi-gpu bat nhung chi co 1 GPU. Chay single-GPU.")
            train(rank=0, world_size=1, args=args)
        elif "RANK" in os.environ:
            # Chay duoi torchrun
            train(rank=int(os.environ["RANK"]),
                  world_size=int(os.environ["WORLD_SIZE"]), args=args)
        else:
            print(f"[INFO] Spawning {world_size} GPU processes...")
            mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=2 train_maskrcnn.py \
#   --data-root .. --gf7-bands 3 --epochs 50 --batch-size 8 \
#   --multi-gpu --amp --scale-lr \
#   --layerwise-lr --warmup-iters 50 --scheduler cosine \
#   --ema --clip-grad-norm 1.0 \
#   --vis-every 10 --vis-num-samples 4
