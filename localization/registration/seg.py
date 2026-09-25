"""Building-probability inference shared by all datasets.

Large rasters: overlapping 500 px tiles with pyramid blending (as M0). Tiles
at the right/bottom border are shifted inwards rather than zero-padded, and
images smaller than one tile are run whole: black padding makes Mask R-CNN
hallucinate large blobs along the padding edge.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_TT = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def _combine(pred, h, w, thr):
    out = torch.zeros((h, w), dtype=torch.float32, device=pred["masks"].device)
    if pred["masks"].numel():
        keep = pred["scores"] >= thr
        if keep.any():
            out = pred["masks"][keep, 0].max(0).values
    return out


@torch.no_grad()
def soft_mask(img: Image.Image, model, device, patch=500, overlap=100, batch=12, thr=0.5) -> np.ndarray:
    W, H = img.size
    if W <= patch and H <= patch:
        pred = model([_TT(img).to(device)])[0]
        return _combine(pred, H, W, thr).cpu().numpy()
    stride = patch - overlap
    xs = list(range(0, max(W - patch, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - patch, 0) + 1, stride)) or [0]
    if xs[-1] + patch < W: xs.append(max(W - patch, 0))
    if ys[-1] + patch < H: ys.append(max(H - patch, 0))
    pw, ph = min(patch, W), min(patch, H)
    wx = 1 - np.abs(np.linspace(-1, 1, pw)); wy = 1 - np.abs(np.linspace(-1, 1, ph))
    wt = torch.from_numpy(np.clip(np.outer(wy, wx), 1e-3, None).astype(np.float32)).to(device)
    acc = torch.zeros((H, W), device=device); cnt = torch.zeros((H, W), device=device)
    coords = [(x, y) for y in ys for x in xs]
    arr = np.asarray(img)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        ts = [_TT(Image.fromarray(arr[y:y + ph, x:x + pw])).to(device) for x, y in chunk]
        preds = model(ts)
        for (x, y), p in zip(chunk, preds):
            acc[y:y + ph, x:x + pw] += _combine(p, ph, pw, thr) * wt
            cnt[y:y + ph, x:x + pw] += wt
    return (acc / cnt.clamp_min(1e-6)).clamp(0, 1).cpu().numpy()
