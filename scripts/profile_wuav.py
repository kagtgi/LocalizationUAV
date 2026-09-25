"""Time each stage of one World-UAV query (debug helper)."""
import sys, time, math, json
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization.registration.structure import query_structure, reference_maps, resample, EnsembleConfig
from localization.registration.fft_search import SearchConfig, search, _templates
from localization.registration.refine import refine
import eval_wuav as E

T = time.time
t = T(); dev = torch.device("cuda"); model = E.load_model("best_model.pth", dev); print("model", T() - t)
prob = cv2.imread("cache/wuav/sat_prob.png", 0).astype(np.float32) / 255
t = T(); pw = resample(prob, 0.3, 0.6); ref = reference_maps(pw, 0.6); print("ref", T() - t, ref.G.shape)
fp = sorted(Path("data/worlduav/Rot/SouthernSuburbs/query/height100_rot90/footage").glob("*.jp*g"))[10]
t = T(); im = Image.open(fp).convert("RGB"); f = 2 * 150 * math.tan(math.radians(15)) / 512 / 0.3
im = im.resize((int(im.width * f), int(im.height * f))); qp = E.soft_mask(im, model, dev); print("seg", T() - t, qp.shape)
t = T(); q = query_structure(resample(qp, 0.3, 0.6), 0.6, ens=EnsembleConfig(dp_tol_m=(0.5, 1.0))); print("qs", T() - t, len(q.pts), q.n_buildings)
cfg = SearchConfig(thetas_deg=tuple(np.arange(0, 360, 5.0)), scales=(0.8, 0.9, 1.0, 1.1, 1.25), sigma_theta_deg=1e6, device="cuda")
t = T(); tp = _templates(q, [(a, s) for a in cfg.thetas_deg for s in cfg.scales], 0.6, True); print("templates", T() - t, tp[0][0].shape)
t = T(); r = search(q, ref, None, None, cfg); torch.cuda.synchronize(); print("search", T() - t)
t = T(); rr = refine(q, ref, r.peaks[0], cfg); print("refine", T() - t)
