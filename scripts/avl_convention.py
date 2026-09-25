"""Find the AnyVisLoc pose convention empirically: which transform projects the
aerial DSM ground points of the scene INTO the image, in front of the camera?"""
import json, sys, numpy as np
from pathlib import Path
sd = Path(sys.argv[1] if len(sys.argv) > 1 else "data/anyvisloc/Scene_10")
refj = json.load(open(next(sd.glob("*_reference.json"))))["modes"]["aerial"]
dsm = np.load(sd / "aerial_dsm.npy").astype(np.float64)
print("dsm", dsm.shape, np.nanpercentile(dsm, [1, 50, 99]))
res = refj["dsm_resolution"][0]; o = refj["dsm_origin_local"]
H, W = dsm.shape[:2]
jj, ii = np.meshgrid(np.arange(0, W, 5), np.arange(0, H, 5))
X = o[0] + jj * res; Y = o[1] + ii * res; Z = dsm[ii, jj] if dsm.ndim == 2 else dsm[ii, jj, 0]
P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1); P = P[np.isfinite(P).all(1)]
for f in sorted(sd.glob("L*_*.npz"))[:40:8]:
    z = np.load(f); K = z["K"]; c2w = z["pose_c2w"].astype(np.float64); w2c = z["pose_w2c"].astype(np.float64)
    h, w = z["image"].shape[:2]
    print(f.stem, "xyz", z["xyz"], "euler", z["euler_deg"], "det", round(np.linalg.det(c2w[:3, :3]), 3))
    for name, M in [("w2c", w2c), ("inv(c2w)", np.linalg.inv(c2w)), ("c2w", c2w)]:
        for zs in (1, -1):
            Q = P.copy(); Q[:, 2] *= zs
            pc = Q @ M[:3, :3].T + M[:3, 3]
            front = pc[:, 2] > 0
            u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]; v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
            inimg = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            print(f"   {name:9s} zsign={zs:+d}: in-image ground pts {inimg.mean()*100:5.1f}%  front {front.mean()*100:5.1f}%")
