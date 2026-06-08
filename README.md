# LocalizationUAV

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c)
![Status](https://img.shields.io/badge/status-research%20reference-success)

Reference implementation of **"Structure-Based UAV Visual Geo-Localization via
Delaunay Triangulation and Ekeland Free-Cone Angle Descriptors"**
([`paper_draft/main.tex`](paper_draft/main.tex)).

Given a single nadir UAV image, the method recovers the drone's 2-D geographic
position by matching **purely geometric building-footprint descriptors** against
a pre-indexed satellite map. It is **appearance-invariant** (no sensitivity to
season, illumination, or sensor) and requires **no scene-specific training** —
the only learned component is a pre-trained Mask R-CNN building segmenter.

```
        OFFLINE  (once per satellite site)        ONLINE  (per UAV image)
        ─────────────────────────────────        ─────────────────────────
        satellite01.tif (9774×26762)              01_0022.JPG
                │                                         │
        500×500 patches @ 100-px stride           yaw + scale + crop → 500×500
                │                                         │
        Mask R-CNN  ◄──── same 500×500 backbone ────►  Mask R-CNN
                │                                         │
        contour → Douglas-Peucker (τ=2px)         contour → Douglas-Peucker
                │                                         │
        Constrained Delaunay Triangulation        Constrained Delaunay Triangulation
                │                                         │
        MFCA / Ekeland angle (D_max=4)            MFCA / Ekeland angle
                │                                         │
        f = (α₁, α₂, e₁, e₂, e₃)                  f = (α₁, α₂, e₁, e₂, e₃)
                │                                         │
        SatelliteDatabase ───────────────────►  K=5 NN retrieval (ℓ₁)
        (scipy KDTree, ℓ₁, leaf 40)                       │
                                                  plurality vote on patch_id
                                                          │
                                                  predicted pixel → lat/lon + error
```

For a guided, **step-by-step** walkthrough (input → output → visualization) see
**[TUTORIAL.md](TUTORIAL.md)**.

---

## Highlights

- **Training-free geometry descriptor.** Each building footprint is triangulated
  and encoded as a 5-D vector `f = (α₁, α₂, e₁, e₂, e₃)`: two interior angles
  (shape) + three sorted Ekeland free-cone angles (inter-building context).
- **One backbone, two branches.** UAV and satellite imagery pass through the
  *same* `segment_batch` at 500×500, so segmentation quality is identical.
- **Fast retrieval.** Satellite descriptors are indexed offline in a
  `scipy.spatial.KDTree` (ℓ₁ metric, leaf 40); each query is K=5 NN per UAV
  triangle followed by a plurality vote over satellite patches.
- **Built-in diagnostics.** Two ready-made figures per query — a 3-panel
  result view and a "GT vs top-100 matches" map (paper Fig. 5(a) style).
- **Verified against the paper.** `scripts/verify_finalization.py` checks the
  geometry/descriptor implementation against the paper's definitions.

---

## Method at a glance

| Step | Paper | Module |
|------|-------|--------|
| 1. Preprocess UAV (yaw, scale, crop → 500×500) | §4.1 | [`localization/preprocess/uav.py`](localization/preprocess/uav.py) |
| 2. Mask R-CNN segmentation + Douglas-Peucker (τ=2px) | §4.2 | [`localization/segmentation/`](localization/segmentation/inference.py) |
| 3. Constrained Delaunay Triangulation (interior-filtered) | §4.3 | [`localization/geometry/triangulation.py`](localization/geometry/triangulation.py) |
| 4. MFCA / Ekeland angle → 5-D descriptor (D_max=4) | §4.4 | [`localization/geometry/ekeland.py`](localization/geometry/ekeland.py), [`descriptor.py`](localization/geometry/descriptor.py) |
| 5a. Patch-based offline KD-Tree index (ℓ₁) | §4.5 | [`localization/database/`](localization/database/builder.py) |
| 5b. K-NN retrieval + plurality vote → position | §4.5 | [`localization/matching/query.py`](localization/matching/query.py) |
| Visualization | — | [`localization/matching/visualize.py`](localization/matching/visualize.py) |

The winning patch's geographic centroid is the predicted UAV position, converted
to lat/lon via the linear bounds in `satellite_coordinates_range.csv`.

---

## Installation

```bash
git clone https://github.com/kagtgi/LocalizationUAV.git
cd LocalizationUAV

# Option A — pip + venv
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

# Option B — conda
conda env create -f environment.yaml   # creates env "uav_env" (Python 3.10)
conda activate uav_env
```

Then fetch the two large assets (both are gitignored):

1. **Mask R-CNN checkpoint** `best_model.pth` (~170 MB), hosted on Google Drive:
   ```bash
   pip install gdown
   gdown 'https://drive.google.com/uc?id=1jlRfOXYU18DcOjWEwNBet1b7FHCIClcB' -O best_model.pth
   ```
2. **UAV-VisLoc dataset** (~17.7 GB) — place under `UAV-VisLoc/` (see [Dataset](#dataset)),
   or run the Kaggle cell in the notebooks to download it automatically.

> A CUDA GPU is recommended for the one-time satellite-database build
> (~25k patches/site). Queries run fine on CPU.

---

## Quick start (notebooks)

Run them in order; each is self-contained (clone → install → data → model →
compute → visualize):

```bash
jupyter notebook notebooks/01_build_satellite_database.ipynb   # build the KD-Tree DB for a site
jupyter notebook notebooks/02_query_uav_localization.ipynb     # localize one UAV image + 2 figures
jupyter notebook notebooks/03_full_pipeline.ipynb              # DB build + 5 random queries, end-to-end
```

| Notebook | Input | Output |
|----------|-------|--------|
| `01_build_satellite_database` | `satellite01.tif` | `outputs/01/satellite01_kdtree.npz` + `_descriptors.csv` |
| `02_query_uav_localization` | one UAV image + DB | predicted lat/lon, error vs GT, `match_top100_*.png`, `gt_vs_top100_*.png` |
| `03_full_pipeline` | a site | DB + 5 per-image figures + a summary table |

---

## Outputs & visualizations

Each query produces two figures under `outputs/{flight_id}/`:

- **`match_top100_<image>.png`** — 3-panel view: full satellite with the 100
  ranked candidate patches (rank-graded markers), a zoom around rank-1, and the
  UAV query image. Rank-1 = red ✕, ground truth = magenta ★.
- **`gt_vs_top100_<image>.png`** — single-panel "GT vs top-100 matches" map
  (paper Fig. 5(a) style): every candidate centroid as a small dot, the ground
  truth as a red ★, and a **yellow ring** on the candidate nearest the GT — a
  recall diagnostic showing whether the correct place is in the top-100.

Generate the Fig. 5(a) map directly:

```python
from localization import draw_gt_vs_topn_centroids

draw_gt_vs_topn_centroids(
    satellite_image_path='UAV-VisLoc/01/satellite01.tif',
    top_n=result.top_n,                # from query_uav(..., top_n=100)
    gt_pixel_xy=gt_pixel,              # optional
    title='Satellite map (Changjiang, 405 m, phi=165 deg)',
    output_path='outputs/01/gt_vs_top100_01_0022.png',
)
```

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

# Offline — run once per site.
df = build_satellite_descriptors(
    tif_path='UAV-VisLoc/01/satellite01.tif',
    model=model, device=device,
    patch_size=500, stride=100, batch_size=4,   # tolerance_px=2.0, max_depth=4 by default
)
db = SatelliteDatabase.from_dataframe(df, parent_tif='satellite01.tif', leaf_size=40)
db.save('outputs/01/satellite01_kdtree.npz')

# Online — per UAV image.
_, img_500, _meta = process_uav('UAV-VisLoc/01/drone/01_0022.JPG', 'UAV-VisLoc/01/01.csv')
uav_desc, _ = extract_patch_descriptors(img_500, model=model, device=device)
result = query_uav(uav_desc, db, k=5, top_n=100)
print(result.patch_id, result.pixel_xy, result.vote_count, result.margin)
```

`import localization` does **not** import torch; lazy `__getattr__` defers heavy
modules, so a consumer can `import localization.database` (only scipy + numpy +
pandas) without paying the torch import cost.

---

## Configuration (paper §4.6 defaults)

Defaults live in [`localization/config.py`](localization/config.py) and as
function arguments.

| Component | Parameter | Default | Where |
|-----------|-----------|---------|-------|
| Segmentation | score threshold | `0.5` | `SegmentationConfig` |
| Segmentation | Douglas-Peucker tolerance | `τ = 2 px` | `SegmentationConfig.douglas_peucker_tolerance_px` |
| Geometry (MFCA) | kernel expansion depth | `D_max = 4` | `GeometryConfig.max_expansion_depth` |
| Index | patch size / stride | `500 / 100` | `IndexConfig` |
| Index | KD-Tree leaf size / metric | `40` / ℓ₁ | `IndexConfig` / `kdtree.py` |
| Retrieval | K nearest neighbours | `5` | `query_uav(k=...)` |

The `D_max ∈ {1, 2, 4, 8}` ablation (paper) is reachable via
`triangle_descriptors_from_polygon(polygon, max_depth=...)` or
`build_satellite_descriptors(..., max_depth=...)`.

---

## Verification

```bash
python scripts/verify_finalization.py
```

Runs offline (no dataset/checkpoint needed) and checks the geometry against the
paper: interior-filtered triangulation, the `D_max` cap, Ekeland-angle
extraction matching **Definition 2** (reflex vertex → 90°, convex → 180°),
monotonicity, and the new Fig. 5(a) plot.

> **Reproducibility note.** The descriptor uses a fixed Douglas-Peucker
> tolerance, interior-filtered (boundary-respecting) triangulation, and a
> `D_max = 4` expansion cap. If you have a KD-Tree database built before these
> were in place, **rebuild it** (re-run notebook 01 / the build cell in
> notebook 03). See [`log.txt`](log.txt) for details.

---

## Repository layout

```text
LocalizationUAV/
  localization/                main package (lazy top-level imports)
    preprocess/uav.py            Step 1 — UAV preprocessing
    segmentation/                Step 2 — Mask R-CNN + contour/Douglas-Peucker
    geometry/                    Steps 3-4 — triangulation, ekeland, descriptor
    database/                    Step 5a — patches, builder, kdtree
    matching/                    Step 5b — query, visualize
    io/                          bounds, dataset loader, CSV export
    config.py, utils.py
  notebooks/
    01_build_satellite_database.ipynb   git clone → KD-Tree .npz
    02_query_uav_localization.ipynb     load DB → query → 2 figures
    03_full_pipeline.ipynb              clone → DB → 5 random samples
  scripts/
    train_maskrcnn.py, test_maskrcnn.py
    verify_finalization.py              offline paper-fidelity checks
  legacy/                      archived non-paper code (see legacy/README.md)
  paper_draft/main.tex         the paper
  log.txt                      finalization / paper-fidelity notes
  UAV-VisLoc/                  dataset scaffold (gitignored data)
  outputs/                     gitignored figures + databases
  TUTORIAL.md, README.md
```

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
11 flights across China, 6,742 drone images, 11 satellite GeoTIFFs, altitudes
400–2,000 m, summer + autumn, ~17.7 GB. Expected on-disk layout (the Kaggle
download produces this directly):

```
UAV-VisLoc/
    satellite_coordinates_range.csv     mapname, LT_lat, LT_lon, RB_lat, RB_lon, region
    01/
        drone/01_0001.JPG ...           UAV images (flight 01)
        satellite01.tif                 9774 × 26762 (Changjiang-20)
        01.csv                          num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2
    02/ ... 11/
```

---

## What's in `legacy/`

Code preserved from earlier experimental tracks but **not** part of the paper's
pipeline (Swin-T mutual learning, brute-force `O(M·N)` matcher, extra distance
metrics). The paper specifies KD-Tree (ℓ₁) + plurality voting, which is what
`localization/database` and `localization/matching` implement. See
[legacy/README.md](legacy/README.md).

---

## Citation

```bibtex
@article{structurebaseduavloc,
  title  = {Structure-Based UAV Visual Geo-Localization via Delaunay
            Triangulation and Ekeland Free-Cone Angle Descriptors},
  note   = {Paper draft in paper_draft/main.tex},
}
```
