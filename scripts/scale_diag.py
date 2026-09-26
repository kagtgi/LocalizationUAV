"""Independent scale diagnosis (calibration site only, uses GT): gradient-magnitude NCC
(cv2.matchTemplate) of the north-up UAV frame vs the satellite around GT over a scale grid.
usage: scale_diag.py SITE N C0"""
import sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_reg as E
from localization.io.dem import DEM

Image.MAX_IMAGE_PIXELS = None
site, n, c0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
G = 1.0                                   # m/px for the diagnosis
root = Path("data/visloc_raw/UAV_VisLoc_dataset")
g = E.site_geo(root, site)
sat = Image.open(root / site / f"satellite{site}.tif")
fl = E.VisLocFlight(site, root)
df = E.load_flight_metadata(fl.metadata_csv)
df = df[df["filename"].str.lower().str.endswith((".jpg", ".jpeg", ".png"))].reset_index(drop=True)
dem = DEM("../data/dem")


def grad(a):
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))


scales = np.exp(np.linspace(np.log(0.2), np.log(1.6), 29))
for i in np.linspace(0, len(df) - 1, n).round().astype(int):
    row = df.iloc[i]
    agl = float(row["height"]) - dem.window_median(float(row["lat"]), float(row["lon"]), 300)
    im = Image.open(fl.drone_image_path(row["filename"])).convert("L")
    gsd0 = c0 * agl / im.width
    gx, gy = E.latlon_to_px_f(float(row["lat"]), float(row["lon"]), g)
    R = 700.0                                                   # search half-size (m)
    hp = int(R / g["gsd"])
    S = np.asarray(sat.crop((int(gx) - hp, int(gy) - hp, int(gx) + hp, int(gy) + hp)).convert("L").resize((int(2 * R / G),) * 2))
    Sg = grad(S)
    res = []
    for s in scales:
        f = gsd0 * s / G
        q = im.resize((max(8, int(im.width * f)), max(8, int(im.height * f)))).rotate(-float(row["Phi1"]), expand=False)
        q = np.asarray(q)
        h, w = q.shape
        cy, cx = h // 2, w // 2
        side = int(min(h, w) / 1.5)                             # inner square avoids rotation corners
        qc = q[cy - side // 2: cy + side // 2, cx - side // 2: cx + side // 2]
        if qc.shape[0] < 16 or qc.shape[0] >= Sg.shape[0]:
            continue
        r = cv2.matchTemplate(Sg, grad(qc), cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        err = np.hypot(loc[0] + qc.shape[1] / 2 - Sg.shape[1] / 2, loc[1] + qc.shape[0] / 2 - Sg.shape[0] / 2) * G
        res.append((mx, s, err))
    res.sort(reverse=True)
    top = res[0]
    print(f"{row['filename']} agl={agl:.0f} gsd0={gsd0:.3f}  best s={top[1]:.3f} ncc={top[0]:.3f} err={top[2]:.0f} m | "
          + " ".join(f"s={s:.2f}:{m:.2f}/{e:.0f}m" for m, s, e in res[:4]))
