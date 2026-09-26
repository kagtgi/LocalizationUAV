"""UAV-VisLoc scale check: north-up UAV frame at c=2tan(HFOV/2) (nominal) and AGL from DEM,
next to the satellite crop of the same metric size at GT. usage: visloc_scale_viz.py SITE N C"""
import sys, math
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_reg as E
from localization.io.dem import DEM

Image.MAX_IMAGE_PIXELS = None
site, n, c = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
root = Path("data/visloc_raw/UAV_VisLoc_dataset")
g = E.site_geo(root, site)
sat = Image.open(root / site / f"satellite{site}.tif")
fl = E.VisLocFlight(site, root)
df = E.load_flight_metadata(fl.metadata_csv)
df = df[df["filename"].str.lower().str.endswith((".jpg", ".jpeg", ".png"))].reset_index(drop=True)
dem = DEM("../data/dem")
out = Path("debug"); out.mkdir(exist_ok=True)
for i in np.linspace(0, len(df) - 1, n).round().astype(int):
    row = df.iloc[i]
    agl = float(row["height"]) - dem.window_median(float(row["lat"]), float(row["lon"]), 300)
    img = Image.open(fl.drone_image_path(row["filename"])).convert("RGB")
    gsd = c * agl / img.width                           # m/px of the raw image
    f = gsd / 0.6                                        # to 0.6 m/px
    img = img.resize((int(img.width * f), int(img.height * f))).rotate(-float(row["Phi1"]), expand=True)
    q = np.asarray(img)[:, :, ::-1]
    gx, gy = E.latlon_to_px_f(float(row["lat"]), float(row["lon"]), g)
    half_m = 0.6 * max(q.shape[:2]) / 2 * 1.3
    hp = int(half_m / g["gsd"])
    crop = sat.crop((int(gx) - hp, int(gy) - hp, int(gx) + hp, int(gy) + hp)).convert("RGB")
    crop = np.asarray(crop.resize((int(2 * half_m / 0.6),) * 2))[:, :, ::-1]
    H = crop.shape[0]; pad = np.zeros_like(crop)
    y0, x0 = (H - q.shape[0]) // 2, (H - q.shape[1]) // 2
    pad[max(y0, 0):max(y0, 0) + min(q.shape[0], H), max(x0, 0):max(x0, 0) + min(q.shape[1], H)] = q[:H, :H]
    cv2.imwrite(str(out / f"scale_{site}_{row['filename']}.jpg"), cv2.resize(np.hstack([pad, crop]), None, fx=0.5, fy=0.5))
    print(row["filename"], "agl", round(agl), "gsd_raw", round(gsd, 3), "footprint_m", round(gsd * Image.open(fl.drone_image_path(row["filename"])).width))
