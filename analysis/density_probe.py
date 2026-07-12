#!/usr/bin/env python
"""Rank UAV-VisLoc sites by building density: segment N evenly-spaced drone
frames per site and count building polygons/frame. High = urban (good for a
building-based method). Writes outputs/paper/density_probe.json. GPU seg.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from localization.preprocess.uav import process_uav
from localization.segmentation.inference import segment_image
from localization.segmentation.model import load_model

DATA = Path("UAV-VisLoc")
MODEL = "best_model.pth"
N_FRAMES = 6
SITES = [f"{i:02d}" for i in range(1, 12)]
OUT = Path("outputs/paper"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL, device=dev, num_classes=2, pretrained=False).to(dev).eval()
    rows = {}
    for s in SITES:
        ddir = DATA / s / "drone"
        csv = DATA / s / f"{s}.csv"
        if not ddir.is_dir() or not csv.exists():
            continue
        frames = sorted(ddir.glob("*.JPG"))
        if not frames:
            continue
        idx = np.linspace(0, len(frames) - 1, min(N_FRAMES, len(frames))).astype(int)
        counts, alts = [], []
        for i in idx:
            fp = frames[int(i)]
            try:
                _, img500, pmeta = process_uav(str(fp), str(csv))
                _m, polys = segment_image(image=img500, model=model, device=dev,
                                          score_threshold=0.5, min_area=50.0, tolerance_px=2.0)
                counts.append(len(polys))
                a = pmeta.get("height")
                if a not in (None, ""):
                    alts.append(float(a))
            except Exception as exc:
                print(f"  {s}/{fp.name}: {exc}", flush=True)
        if counts:
            rows[s] = {"mean_polys": round(float(np.mean(counts)), 1),
                       "median_polys": float(np.median(counts)),
                       "min": int(np.min(counts)), "max": int(np.max(counts)),
                       "n_probed": len(counts),
                       "median_alt_m": round(float(np.median(alts)), 0) if alts else None}
            print(f"site {s}: mean={rows[s]['mean_polys']:.1f} polys/frame "
                  f"median={rows[s]['median_polys']:.0f} alt~{rows[s]['median_alt_m']}m", flush=True)
    ranked = sorted(rows.items(), key=lambda kv: kv[1]["mean_polys"], reverse=True)
    print("\n=== building-density ranking (urban -> rural) ===", flush=True)
    for s, v in ranked:
        print(f"  {s}: {v['mean_polys']:.1f} polys/frame", flush=True)
    (OUT / "density_probe.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[probe] wrote {OUT/'density_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
