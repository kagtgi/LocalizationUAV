# LocalizationUAV

Reference implementation of **"Structure-Based UAV Visual Geo-Localization
via Delaunay Triangulation and Ekeland Free-Cone Angle Descriptors"**
([`paper_draft/main.tex`](paper_draft/main.tex)).

Given a single UAV nadir image, the method recovers the drone's 2-D
geographic position by matching purely geometric building-footprint
descriptors against a pre-indexed satellite map. **No scene-specific
training**: the only learned component is a pre-trained Mask R-CNN.

```
Offline (per satellite site)            Online (per UAV image)
satellite01.tif                         01_0022.JPG
        |                                       |
   500x500 patches                       yaw + scale + crop
        |                                       |
   Mask R-CNN  <-- same 500x500 backbone -->   Mask R-CNN
        |                                       |
  contour -> CDT                          contour -> CDT
        |                                       |
  MFCA descriptor                          MFCA descriptor
  f = (a1, a2, e1, e2, e3)                f = (a1, a2, e1, e2, e3)
        |                                       |
  SatelliteDatabase                       query (K=5 NN, ell_1)
  (scipy KDTree, ell_1)  <-------------- plurality vote on patch_id
                                                 |
                                       predicted pixel + lat/lon
```

For a **full step-by-step walkthrough**, see **[TUTORIAL.md](TUTORIAL.md)**.

---

## Method at a glance

| Step | Paper | Module |
|------|-------|--------|
| 1. Preprocess UAV (yaw, scale, crop → 500×500) | §4.1 | [`localization/preprocess`](localization/preprocess/uav.py) |
| 2. Mask R-CNN building segmentation | §4.2 | [`localization/segmentation`](localization/segmentation/inference.py) |
| 3. Constrained Delaunay Triangulation | §4.3 | [`localization/geometry/triangulation.py`](localization/geometry/triangulation.py) |
| 4. MFCA / Ekeland angle → 5-D descriptor | §4.4 | [`localization/geometry/ekeland.py`](localization/geometry/ekeland.py), [`descriptor.py`](localization/geometry/descriptor.py) |
| 5a. Patch-based offline KD-Tree index (ℓ₁) | §4.5 | [`localization/database`](localization/database/) |
| 5b. K-NN retrieval + plurality vote | §4.5 | [`localization/matching/query.py`](localization/matching/query.py) |

Both UAV and satellite go through the **same** `segment_batch` function at
the same 500×500 resolution, so segmentation quality is identical on both
branches. The satellite map is tiled into 500-px patches at 100-px stride;
each triangle vote on the winning patch is converted to a geographic
position via the linear bounds in `satellite_coordinates_range.csv`.

---

## Repository layout

```text
LocalizationUAV/
  localization/                main package (lazy top-level imports)
    preprocess/uav.py            Step 1
    segmentation/                Step 2 (model + inference + contours)
    geometry/                    Steps 3-4 (triangulation, ekeland, descriptor)
    database/                    Step 5a (patches, builder, kdtree)
    matching/                    Step 5b (query, visualize)
    io/                          bounds, dataset loader, CSV export
    config.py, utils.py
  notebooks/
    01_build_satellite_database.ipynb   from git clone -> KDTree .npz
    02_query_uav_localization.ipynb     load DB -> query -> red-X figure
  scripts/                     train_maskrcnn.py, test_maskrcnn.py
  legacy/                      archived non-paper code (see legacy/README.md)
  paper_draft/main.tex         the paper
  UAV-VisLoc/                  dataset scaffold (gitignored data)
  outputs/                     gitignored notebook outputs
  TUTORIAL.md, README.md
```

---

## Quick start

```bash
git clone https://github.com/kagtgi/LocalizationUAV.git && cd LocalizationUAV
pip install -r requirements.txt
# Download the trained Mask R-CNN checkpoint (best_model.pth) from
#   https://drive.google.com/file/d/1jlRfOXYU18DcOjWEwNBet1b7FHCIClcB/view
# and place it at the repo root. The notebooks also do this automatically
# via `gdown` if the file is missing.
# Place UAV-VisLoc/ data per TUTORIAL.md §3 (or run the Kaggle cell in nb1).
jupyter notebook notebooks/01_build_satellite_database.ipynb   # build DB
jupyter notebook notebooks/02_query_uav_localization.ipynb     # localize 01_0022.JPG
```

> **Note on `best_model.pth`**: the trained Mask R-CNN checkpoint is too
> large for git; it is hosted on Google Drive at the link above. The
> notebooks include a one-line `gdown` cell that pulls it automatically.

For automated Kaggle download, environment setup, expected runtimes, and
troubleshooting, see **[TUTORIAL.md](TUTORIAL.md)**.

---

## Programmatic use

```python
import torch
from localization import (
    load_model, process_uav,
    build_satellite_descriptors, SatelliteDatabase, query_uav,
)
from localization.database.builder import extract_patch_descriptors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = load_model('best_model.pth', device=device, num_classes=2,
                    pretrained=False).to(device).eval()

# Offline (run once per site)
df = build_satellite_descriptors(
    tif_path='UAV-VisLoc/01/satellite01.tif',
    model=model, device=device,
    patch_size=500, stride=100, batch_size=4,
)
db = SatelliteDatabase.from_dataframe(df, parent_tif='satellite01.tif', leaf_size=40)
db.save('outputs/01/satellite01_kdtree.npz')

# Online (per UAV image)
_, img_500, _meta = process_uav(
    'UAV-VisLoc/01/drone/01_0022.JPG',
    'UAV-VisLoc/01/01.csv',
)
uav_desc, _ = extract_patch_descriptors(img_500, model=model, device=device)
result = query_uav(uav_desc, db, k=5)
print(result.patch_id, result.pixel_xy, result.vote_count, result.margin)
```

Importing `localization` does not pull in torch; lazy `__getattr__` defers
heavy modules until first use, so a downstream consumer can `import
localization.database` (only scipy + numpy + pandas) without paying the
torch import cost.

---

## Training the Mask R-CNN

The shipped `best_model.pth` was trained on the GF-7 building dataset; to
retrain see [scripts/train_maskrcnn.py](scripts/train_maskrcnn.py):

```bash
python scripts/train_maskrcnn.py \
  --data-root .. --gf7-bands 3 \
  --epochs 20 --batch-size 2 \
  --output-dir checkpoints/maskrcnn_gf7_3band
```

3-band RGB and 4-band (RGB + NIR) inputs are both supported.

---

## Dataset

[UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc)
([Kaggle](https://www.kaggle.com/datasets/hailong1610/uav-visloc-dataset)):
11 flights across China, 6,742 drone images, 11 satellite GeoTIFFs,
altitudes 400-2,000 m, summer + autumn. ~17.7 GB.

The expected on-disk layout (the Kaggle download produces this directly):

```
UAV-VisLoc/
    satellite_coordinates_range.csv     mapname,LT_lat_map,LT_lon_map,RB_lat_map,RB_lon_map,region
    01/
        drone/01_0001.JPG ...           817 UAV images (flight 01)
        satellite01.tif                 9774 x 26762 (Changjiang-20)
        01.csv                          num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2
    02/ ... 11/
```

---

## What's in `legacy/`

Code preserved from earlier experimental tracks but **not** part of the
paper's pipeline:

- `mutual_learning_swin.py` — Swin-T + ResNet-50 dual-Mask-R-CNN mutual
  learning experiment
- `brute_force_matcher.py` — original `O(M·N)` triangle-pair matcher
- `extra_distance_metrics.py` — `complex` / `geodesic` / `circular_mean` /
  `von_mises` variants

The paper specifies KD-Tree (`scipy.spatial.KDTree`, p=1 / ℓ₁) + plurality
voting on patch IDs, which is what `localization/database` and
`localization/matching` implement. See [legacy/README.md](legacy/README.md).

---

## Citation

```
@article{structurebaseduavloc,
  title  = {Structure-Based UAV Visual Geo-Localization via Delaunay
            Triangulation and Ekeland Free-Cone Angle Descriptors},
  note   = {Paper draft in paper_draft/main.tex},
}
```
