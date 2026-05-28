# Tutorial: End-to-End Localization Pipeline

This tutorial walks through the full pipeline of *Structure-Based UAV Visual
Geo-Localization via Delaunay Triangulation and Ekeland Free-Cone Angle
Descriptors* on the [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc)
benchmark.

**Flow**: satellite GeoTIFF → KD-Tree database (offline) → UAV image →
preprocessed 500×500 image → 5-D triangle descriptors → KD-Tree query →
plurality vote → predicted geographic position.

```
Offline (run once per site)              Online (run per UAV query)
satellite01.tif (9774×26762)             01_0022.JPG (3976×2652)
        |                                        |
   iter_patches (500/100)              process_uav (yaw/scale/crop)
        |                                        |
 segment_batch (500x500 Mask R-CNN) ----segment_batch (500x500 Mask R-CNN)
        |                                        |
   polygons -> CDT -> MFCA              polygons -> CDT -> MFCA
        |                                        |
  5-D descriptors per patch              5-D descriptors per UAV triangle
        |                                        |
  SatelliteDatabase (KD-tree, ell_1) <-- query (K=5 NN per UAV triangle)
                                                 |
                                       plurality vote on patch_id
                                                 |
                                        predicted pixel/lat/lon + red X
```

The same `segment_batch` function runs on both branches at 500×500, so
segmentation quality is identical for UAV and satellite inputs.

---

## 0. Prerequisites

- **Python 3.10+**
- **GPU recommended** (CPU works for queries but the satellite DB build will
  be slow — ~25k patches per site)
- **Disk**: ~20 GB free (UAV-VisLoc is ~17.7 GB; DB outputs are tiny)
- **Kaggle API token** (a `kaggle.json` file) if you want to use the
  automated download in the notebooks
- The trained Mask R-CNN checkpoint **`best_model.pth`** — bundled in the
  `building-segment` Kaggle dataset

---

## 1. Clone the repository

```bash
git clone https://github.com/kagtgi/LocalizationUAV.git
cd LocalizationUAV
```

---

## 2. Install dependencies

```bash
# Option A: pip
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yaml
conda activate uav_env
```

---

## 3. Get the data and the Mask R-CNN checkpoint

The pretrained Mask R-CNN weights (`best_model.pth`, ~170 MB) are too large
for git, so they live on Google Drive:

> **Download:** https://drive.google.com/file/d/1jlRfOXYU18DcOjWEwNBet1b7FHCIClcB/view

### Option A — Automatic (inside the notebook)

Both notebooks have a `gdown` cell that fetches `best_model.pth` if it's
missing, and a Kaggle cell that pulls the UAV-VisLoc dataset (using your
`kaggle.json`) and unzips it in place. Drag `kaggle.json` into the working
directory and run the relevant cells.

### Option B — Manual

```bash
# Mask R-CNN checkpoint (~170 MB)
pip install --quiet gdown
gdown 'https://drive.google.com/uc?id=1jlRfOXYU18DcOjWEwNBet1b7FHCIClcB' -O best_model.pth
```

Place files so the layout looks like:

```
LocalizationUAV/
    best_model.pth                          # Mask R-CNN checkpoint (from Google Drive)
    UAV-VisLoc/
        satellite_coordinates_range.csv     # mapname, LT/RB lat/lon (per the user's preview)
        01/
            drone/01_0001.JPG ... 01_0817.JPG
            satellite01.tif                 # 9774 x 26762 (Changjiang-20)
            01.csv                          # per-image GPS + Omega/Kappa/Phi1/Phi2
        02/ ... 11/
```

The empty per-flight folders are already scaffolded in the repo (gitignored
data files are added on top).

---

## 4. Build the satellite database — `notebooks/01_build_satellite_database.ipynb`

Launch Jupyter and open notebook 1. Each cell is grouped under a numbered
heading. What each does:

| Cell group | Module called | What happens |
|---|---|---|
| 1. Clone repo | `git clone` | One-time setup on Colab/Kaggle |
| 2. pip install | `pip install -r requirements.txt` | Install scipy/torch/opencv/skimage/shapely/etc. |
| 3. Kaggle setup + download | `kaggle datasets download ...` | Pull `building-segment` (model) and `uav-visloc-dataset` |
| 4. Paths | — | Defines `DATA_ROOT`, `FLIGHT_ID='01'`, `PATCH_SIZE=500`, `STRIDE=100`, `INFERENCE_BATCH_SIZE=4` |
| 5. Load Mask R-CNN | [`localization.load_model`](localization/segmentation/model.py) | `ResNet-50-FPN`, 2 classes, eval mode, on GPU if available |
| 6. Build descriptors | [`build_satellite_descriptors`](localization/database/builder.py) | Sliding window over `satellite01.tif` (94 × 263 ≈ **24,722 patches**), batched Mask R-CNN inference, CDT, MFCA → DataFrame `[patch_id, parent_tif, top_left_x, top_left_y, centroid_x, centroid_y, alpha1, alpha2, e1, e2, e3]` |
| 7. Build + save DB | [`SatelliteDatabase.from_dataframe`](localization/database/kdtree.py), `.save(...)` | `scipy.spatial.KDTree(p=1)`, leaf=40. Saves to `outputs/01/satellite01_kdtree.npz` |
| 8. Sanity overlay | matplotlib | Random 2,000 centroids plotted on a downsampled satellite preview |
| 9. Round-trip | `SatelliteDatabase.load` | Confirms save/load preserves all arrays |

### What to expect

- **Runtime**: ~5-10 min on a single recent GPU; ~30-60 min on CPU. Inference
  batching (`INFERENCE_BATCH_SIZE=4`) is the dominant speedup. Increase to 8
  if you have GPU memory headroom; decrease to 1 if you run on CPU.
- **Descriptors**: paper §4.4 suggests ~600k per site. You should see a
  number on this order of magnitude in step 6.
- **Database file size**: ~15 MB compressed on disk (matches paper §4.4 of
  ~12 MB for descriptors + a few MB for centroids/patch IDs).
- **Sanity overlay**: yellow dots should cluster on building rooftops in the
  preview.

### Outputs

- `outputs/01/satellite01_descriptors.csv` — raw per-triangle descriptors
- `outputs/01/satellite01_kdtree.npz` — packed KD-Tree database (the file
  notebook 2 loads)

---

### Want everything in one notebook? Use `notebooks/03_full_pipeline.ipynb`

`03_full_pipeline.ipynb` combines DB build + 5 random UAV queries in a
single end-to-end run (clone → install → Kaggle data → gdown checkpoint →
build DB → pick 5 random images → render the top-100 candidate map for
each). It is the fastest way to see the whole pipeline working on flight 01.

## 5. Query a UAV image — `notebooks/02_query_uav_localization.ipynb`

Now the online side. Default demo: `UAV-VisLoc/01/drone/01_0022.JPG`.

| Cell group | Module called | What happens |
|---|---|---|
| 1. Clone & install (skippable) | `git clone`, `pip install` | Re-runs setup if running notebook 2 standalone |
| 2. Kaggle setup (skippable) | `kaggle datasets download` | Re-downloads data if not already present |
| 3. Paths + imports | — | Asserts `DB_NPZ_PATH` exists |
| 4. Load model + DB | [`load_model`](localization/segmentation/model.py), [`SatelliteDatabase.load`](localization/database/kdtree.py) | DB is loaded into memory; KD-Tree built lazily on first query |
| 5. Step 1 — Preprocess UAV | [`process_uav`](localization/preprocess/uav.py) | Reads `01.csv` for `Phi1` (yaw), `Kappa` (roll), `Omega` (pitch), `height`; rotates to north-up; scales by altitude; central crops; resizes to **500×500** |
| 6. Steps 2-4 — extract UAV descriptors | [`extract_patch_descriptors`](localization/database/builder.py) | Same `segment_batch` (batch=1) → polygons → CDT → MFCA → 5-D descriptors |
| 7. Step 5b — query | [`query_uav`](localization/matching/query.py) | K=5 NN per UAV triangle under ell_1; plurality vote on `patch_id`; predicted pixel = pre-computed centroid of the winning patch |
| 8. Pixel → lat/lon, GT comparison | [`pixel_to_latlon`](localization/io/bounds.py), [`pixel_offset_to_meters`](localization/io/bounds.py) | Loads `satellite_coordinates_range.csv` and `01.csv` to compute ground-truth offset in meters |
| 9. Visualize top-100 | [`render_top_n_result`](localization/matching/visualize.py) | 3-panel figure: full `satellite01.tif` with **all 100 ranked candidate patches** drawn as numbered, color-graded markers (rank 1 = bright red, rank 100 = cool purple); zoom around rank-1; UAV view. GT marker (magenta ★) and yellow error connector overlaid when bounds are available |

### What to expect

- **Runtime**: ~1-2 s per query on a GPU (model forward pass dominates).
  KD-Tree query is ~10 ms for ~600k descriptors with K=5.
- **Margin**: difference between the winning patch's votes and the
  runner-up's. A healthy margin is double-digit on a building-dense site;
  near-zero on sparse-building sites means the prediction is brittle.
- **Offset to GT**: paper expects mean error ~10 m on Changjiang-20 in the
  ideal case. The position resolution is bounded by the patch stride
  (100 px × 0.3 m/px = 30 m), per paper §4.5b limitations.

### Outputs

- `outputs/01/01_0022_preprocessed.jpg` — the 500×500 yaw-aligned UAV image
- `outputs/01/match_01_0022.png` — the 2-panel result figure

---

## 6. Try a different UAV image or site

In notebook 2:

```python
UAV_IMAGE_NAME = '01_0050.JPG'   # any image in UAV-VisLoc/01/drone/
```

To target another site, change `FLIGHT_ID` in **both** notebooks and re-run
notebook 1 first to build that site's database:

```python
FLIGHT_ID = '03'                  # Taizhou-1
```

The database NPZ filenames are site-specific (`satellite{FLIGHT_ID}_kdtree.npz`)
so you can keep multiple sites side by side.

---

## 7. Programmatic API

Bypass the notebooks entirely:

```python
import torch
from localization import (
    load_model, process_uav,
    build_satellite_descriptors, SatelliteDatabase, query_uav,
)
from localization.database.builder import extract_patch_descriptors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = load_model('best_model.pth', device=device, num_classes=2, pretrained=False).to(device).eval()

# Offline: one-time DB build per site.
df = build_satellite_descriptors(
    tif_path='UAV-VisLoc/01/satellite01.tif',
    model=model, device=device,
    patch_size=500, stride=100, batch_size=4,
)
db = SatelliteDatabase.from_dataframe(df, parent_tif='satellite01.tif', leaf_size=40)
db.save('outputs/01/satellite01_kdtree.npz')

# Online: preprocess + descriptor + query.
_, img_500, _meta = process_uav('UAV-VisLoc/01/drone/01_0022.JPG', 'UAV-VisLoc/01/01.csv')
uav_desc, _ = extract_patch_descriptors(img_500, model=model, device=device)
result = query_uav(uav_desc, db, k=5)
print(result.patch_id, result.pixel_xy, result.vote_count, result.margin)
```

The top-level `localization` package uses lazy imports — `import
localization` does **not** import torch on its own; only attribute accesses
that need it (`load_model`, `process_uav`, `build_satellite_descriptors`)
trigger the torch import.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AssertionError: Missing satellite TIF` | UAV-VisLoc not downloaded yet | Run the Kaggle cell or place files manually per §3 |
| `AssertionError: Missing Mask R-CNN checkpoint` | `best_model.pth` not at repo root | Run the `gdown` cell in notebook 1 (or `pip install gdown && gdown 'https://drive.google.com/uc?id=1jlRfOXYU18DcOjWEwNBet1b7FHCIClcB' -O best_model.pth`) |
| `assert uav_descriptors.shape[0] > 0, 'No buildings segmented...'` | UAV image is over open terrain (desert/farmland) | Pick another image with visible buildings; the method does not work without buildings (paper §4.7 limitations) |
| CUDA OOM during DB build | `INFERENCE_BATCH_SIZE` too large for your GPU | Lower `INFERENCE_BATCH_SIZE` to 2 or 1 in notebook 1, cell 4 |
| `RuntimeError` from `np.unique` on patch IDs | Empty descriptors DataFrame | Check `descriptors_df['patch_id'].nunique()` is > 0; if 0, the model produced no buildings — try a different site or lower `score_threshold` |
| Bounds CSV not found | `satellite_coordinates_range.csv` missing or named differently | Place at `UAV-VisLoc/satellite_coordinates_range.csv` (the loader also accepts a stray space variant) |
| `ModuleNotFoundError: torch` | Dependencies not installed | `pip install -r requirements.txt` |
| Notebook 2 fails on `assert DB_NPZ_PATH.exists()` | Notebook 1 was not run | Run notebook 1 to completion first |

---

## 9. Where each paper step lives

| Paper step | Code path |
|---|---|
| §4.1 — UAV preprocess | [`localization/preprocess/uav.py`](localization/preprocess/uav.py) |
| §4.2 — Mask R-CNN segmentation | [`localization/segmentation/model.py`](localization/segmentation/model.py), [`inference.py`](localization/segmentation/inference.py), [`contours.py`](localization/segmentation/contours.py) |
| §4.3 — CDT | [`localization/geometry/triangulation.py`](localization/geometry/triangulation.py) |
| §4.4 — MFCA + 5-D descriptor | [`localization/geometry/ekeland.py`](localization/geometry/ekeland.py), [`descriptor.py`](localization/geometry/descriptor.py) |
| §4.5a — Offline indexing | [`localization/database/patches.py`](localization/database/patches.py), [`builder.py`](localization/database/builder.py), [`kdtree.py`](localization/database/kdtree.py) |
| §4.5b — Online query + plurality vote | [`localization/matching/query.py`](localization/matching/query.py) |
| §4.6 — Hyperparameters | [`localization/config.py`](localization/config.py) |
| Visualization | [`localization/matching/visualize.py`](localization/matching/visualize.py), [`localization/geometry/visualize.py`](localization/geometry/visualize.py) |

The full paper is in [`paper_draft/main.tex`](paper_draft/main.tex).

Archived experimental code (not part of the paper) lives in
[`legacy/`](legacy/README.md).
