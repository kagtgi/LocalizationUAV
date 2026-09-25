"""Save side-by-side debug panels: query RGB | query prob | satellite crop | satellite prob."""
import sys, math, json
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_wuav as E

out = Path("debug"); out.mkdir(exist_ok=True)
dev = torch.device("cuda"); model = E.load_model("best_model.pth", dev)
region = Path("data/worlduav/Rot/SouthernSuburbs")
db = E.read_db(region); img, geo, cents = E.mosaic(region, db) if not Path("cache/wuav/sat_rgb.jpg").exists() else (None, None, None)
sat = cv2.imread("cache/wuav/sat_rgb.jpg"); sp = cv2.imread("cache/wuav/sat_prob.png", 0)
cv2.imwrite(str(out / "wuav_sat_small.jpg"), cv2.resize(np.hstack([sat, cv2.cvtColor(cv2.resize(sp, sat.shape[1::-1]), cv2.COLOR_GRAY2BGR)]), None, fx=0.35, fy=0.35))
for name in ["height100_rot0", "height150_rot90"]:
    fp = sorted((region / "query" / name / "footage").glob("*.jp*g"))[10]
    im = Image.open(fp).convert("RGB")
    for scale_note, f in [("seg03", 2 * 150 * math.tan(math.radians(15)) / 512 / 0.3), ("native", 1.0)]:
        imr = im.resize((int(im.width * f), int(im.height * f)))
        p = E.soft_mask(imr, model, dev)
        a = np.asarray(imr)[:, :, ::-1]
        b = cv2.cvtColor((p * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panel = np.hstack([a, b])
        cv2.imwrite(str(out / f"wuav_{name}_{scale_note}.jpg"), cv2.resize(panel, None, fx=512 / panel.shape[0], fy=512 / panel.shape[0]))
print("ok")
