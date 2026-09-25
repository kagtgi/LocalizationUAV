"""World-UAV scale/orientation check: rot0 query (assumed GSD) next to the mosaic crop at GT."""
import sys, json, math
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_wuav as E

region = Path("data/worlduav/Rot/SouthernSuburbs")
db = E.read_db(region); img, geo, cents = E.mosaic(region, db)
mos = img[:, :, ::-1]
for name in ["height100_rot0", "height150_rot0"]:
    cam = json.load(open(region / "query" / name / f"{name}.json"))
    h = int(name.split("_")[0][6:])
    for fi in (40, 60):
        cf = cam["cameraFrames"][fi]
        q = cv2.imread(str(sorted((region / "query" / name / "footage").glob("*.jp*g"))[fi]))
        gsd_q = 2 * h * math.tan(math.radians(cf["fovVertical"] / 2)) / cam["height"]
        f = gsd_q / geo["gsd"]
        qr = cv2.resize(q, None, fx=f, fy=f)
        gx, gy = E.ll_to_px(cf["coordinate"]["latitude"], cf["coordinate"]["longitude"], geo)
        n = 400
        pad = cv2.copyMakeBorder(mos, n, n, n, n, cv2.BORDER_CONSTANT)
        crop = pad[int(gy) + n - n // 2: int(gy) + n + n // 2, int(gx) + n - n // 2: int(gx) + n + n // 2].copy()
        c = n // 2; hh, ww = qr.shape[:2]
        cv2.rectangle(crop, (c - ww // 2, c - hh // 2), (c + ww // 2, c + hh // 2), (0, 0, 255), 2)
        qpad = np.zeros_like(crop); qpad[c - hh // 2: c - hh // 2 + hh, c - ww // 2: c - ww // 2 + ww] = qr[: min(hh, n), : min(ww, n)]
        cv2.imwrite(f"debug/wv_{name}_{fi}.jpg", np.hstack([qpad, crop]))
        print(name, fi, "gsd_q", round(gsd_q, 3), "mosaic gsd", round(geo["gsd"], 3), "query px on mosaic", qr.shape[:2])

# --- segmentation panels: query prob (at 0.3 m) vs mosaic prob crop at GT ---
import torch
dev = torch.device("cuda"); model = E.load_model("best_model.pth", dev)
sp = cv2.imread("cache/wuav/sat_prob.png", 0)          # at 0.3 m/px
k = geo["gsd"] / 0.3
for name in ["height150_rot0"]:
    cam = json.load(open(region / "query" / name / f"{name}.json"))
    for fi in (40, 60):
        cf = cam["cameraFrames"][fi]
        fp = sorted((region / "query" / name / "footage").glob("*.jp*g"))[fi]
        from PIL import Image
        im = Image.open(fp).convert("RGB")
        gsd_q = 2 * 150 * math.tan(math.radians(15)) / 512
        for tag, f in (("seg03", gsd_q / 0.3), ("native", 1.0)):
            imr = im.resize((int(512 * f), int(512 * f)))
            p = (E.soft_mask(imr, model, dev) * 255).astype(np.uint8)
            p = cv2.resize(p, (int(512 * gsd_q / 0.3),) * 2)
            gx, gy = E.ll_to_px(cf["coordinate"]["latitude"], cf["coordinate"]["longitude"], geo)
            cx, cy = int(gx * k), int(gy * k); n = p.shape[0]
            crop = sp[cy - n // 2: cy + n // 2, cx - n // 2: cx + n // 2]
            crop = cv2.resize(crop, (n, n))
            cv2.imwrite(f"debug/wvseg_{name}_{fi}_{tag}.jpg", cv2.resize(np.hstack([p, crop]), None, fx=2, fy=2))
            print(tag, fi, "query building frac", round((p > 127).mean(), 3), "sat frac", round((crop > 127).mean(), 3))
