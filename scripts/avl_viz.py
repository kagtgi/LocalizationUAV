"""Overlay check for AnyVisLoc: rectified ortho vs reference map at GT, both at 0.3 m/px.
usage: python scripts/avl_viz.py Scene_10 L10_0134 L10_0268"""
import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_avl import rectify, off_nadir_deg

sd = Path("data/anyvisloc") / sys.argv[1]
for mode in ("satellite", "aerial"):
    refj = json.load(open(next(sd.glob("*_reference.json"))))["modes"][mode]
    res = refj["map_resolution"][0]; ox, oy = refj["map_origin_local"]
    m = cv2.imread(str(sd / refj["map_path"]))
    f = res / 0.3
    m = cv2.resize(m, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    for sid in sys.argv[2:]:
        z = np.load(sd / f"{sid}.npz")
        c2w = z["pose_c2w"].astype(np.float64); xyz = z["xyz"]
        ortho, ok, h = rectify(z["image"], z["K"].astype(np.float64), z["dist"].astype(np.float64), c2w, 0.0, 0.3, 100.0)
        n = ortho.shape[0]; c = n // 2
        gx, gy = (xyz[0] - ox) / 0.3, (xyz[1] - oy) / 0.3
        pad = cv2.copyMakeBorder(m, n, n, n, n, cv2.BORDER_CONSTANT)
        cx, cy = int(round(gx)) + n, int(round(gy)) + n
        crop = pad[cy - c: cy - c + n, cx - c: cx - c + n]
        blend = cv2.addWeighted(crop, 0.5, ortho[:, :, ::-1], 0.5, 0)
        cv2.imwrite(f"debug/ov_{mode}_{sid}.jpg", np.hstack([ortho[:, :, ::-1], crop, blend]))
        print(mode, sid, "off-nadir", round(off_nadir_deg(c2w), 1), "h", round(float(h), 1), "gt px", round(gx), round(gy), "map", m.shape)
