# UAV_nonGPS

UAV_nonGPS is a notebook-first, modular pipeline for non-GPS UAV localization using geometric features extracted from UAV and satellite imagery.

## What This Project Does

1. Segment buildings/obstacles from satellite and UAV images using a Mask R-CNN model.
2. Convert segmented regions to polygons, triangulate them, and compute Ekeland-based geometric descriptors.
3. Match UAV triangle features against satellite triangle features.
4. Visualize top candidate matches on both UAV and satellite images.

## Project Layout

```text
UAV_nonGPS/
  config.py                  # Runtime/data path configs (dataclasses + legacy config)
  model.py                   # Mask R-CNN creation and checkpoint loading
  dataset.py                 # UAV image pose correction and dataset/dataloader utilities
  run.ipynb                  # Main end-to-end notebook entry point
  requirements.txt           # pip environment
  environment.yaml           # conda environment

  geometry/
    triangulation.py         # Triangulation graph + expansion helpers
    ekeland.py               # Ekeland angle analyzer and computation

  segmentation/
    contours.py              # Contour/polygon extraction + visualization helpers
    pipeline.py              # Segmentation pipeline orchestration

  satellite/
    preprocess.py            # Satellite/UAV feature extraction pipelines + facades

  matching/
    features.py              # Feature extraction, distance metrics, matching, visualization
    pipeline.py              # End-to-end localization orchestrator class

  io/
    export.py                # CSV export helpers for extracted triangle features
```

## Environment Setup

Use either Conda (recommended) or pip.

### Option 1: Conda

```bash
cd UAV_nonGPS
conda env create -f environment.yaml
conda activate uav_env
```

### Option 2: pip

```bash
cd UAV_nonGPS
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Required Inputs

- Trained checkpoint: `../best_model.pth` (or update path in notebook).
- Dataset root folder containing flight folders (e.g. `UAV_nonGPS_dataset/01/...`).
- Satellite bounds file: `UAV_nonGPS_dataset/satellite_coordinates_range.csv`.

## How To Run

Primary entry point: `run.ipynb`.

1. Open `UAV_nonGPS/run.ipynb` in Jupyter.
2. Set paths in the configuration cell:
   - `DATA_ROOT`
   - `FLIGHT_ID`
   - `MODEL_PATH`
   - input UAV/satellite image paths.
3. Run cells in order.

Notebook flow:

1. Add repository root to `sys.path`.
2. Import pipeline modules/classes.
3. Load model with `load_model(...)`.
4. Build satellite feature CSV via `process_satellite_as_drone(...)`.
5. Build rotated UAV feature CSV via `complete_segmentation_demo_uav_with_rotation(...)`.
6. Read UAV ground-truth metadata and satellite bounds.
7. Run OOP matching (`FeatureExtractor` + `DistanceMetric` + `Matcher`).
8. Visualize top matches with `MatchVisualizer.render(...)`.

## Train Mask R-CNN (PyTorch)

The repository now includes `train_maskrcnn.py` for end-to-end Mask R-CNN training with:
- `GF-7 Building (3Bands)` or `GF-7 Building (4Bands)` (`Train/Val` split),
- optional extra training patches from `Images and Shpfiles`.

### 1) Train on GF-7 (3Bands)

```bash
cd UAV_nonGPS
python train_maskrcnn.py \
  --data-root .. \
  --gf7-bands 3 \
  --epochs 20 \
  --batch-size 2 \
  --output-dir checkpoints/maskrcnn_gf7_3band
```

### 2) Train on GF-7 (4Bands)

```bash
cd UAV_nonGPS
python train_maskrcnn.py \
  --data-root .. \
  --gf7-bands 4 \
  --epochs 20 \
  --batch-size 2 \
  --output-dir checkpoints/maskrcnn_gf7_4band
```

### 3) Train on GF-7 (4Bands) + Images and Shpfiles

```bash
cd UAV_nonGPS
python train_maskrcnn.py \
  --data-root .. \
  --gf7-bands 4 \
  --include-images-shp \
  --images-shp-root "../Images and Shpfiles" \
  --extra-patch-size 512 \
  --extra-patch-stride 384 \
  --extra-min-fg-ratio 0.002 \
  --extra-bg-keep-prob 0.03 \
  --epochs 25 \
  --batch-size 2 \
  --output-dir checkpoints/maskrcnn_gf7_4band_plus_areas
```

### Notes

- Labels are expected as binary masks (`0/255`), and are converted to connected-component instances for Mask R-CNN targets.
- For 4-band training, `model.py` automatically adapts the first backbone conv layer to 4 input channels.
- Key outputs:
  - `best_model.pth`
  - `checkpoint_epoch_XXX.pth`
  - `history.json`
  - `train_config.json`

## Main Classes and Important Functions in `run.ipynb` Flow

### Model and Data Preparation

- `load_model` (`model.py`): loads Mask R-CNN checkpoint and returns eval-ready model.
- `process_uav` (`dataset.py`): applies UAV pose normalization (yaw/roll/pitch handling, scaling/cropping) and returns rotated image used for matching visualization.

### Feature Extraction Pipelines

- `process_satellite_as_drone` (`satellite/preprocess.py`):
  - segments large satellite TIFF via sliding window,
  - extracts contours/polygons,
  - triangulates polygons,
  - computes Ekeland features,
  - exports satellite CSV.
- `complete_segmentation_demo_uav_with_rotation` (`satellite/preprocess.py`):
  - rotates UAV image to north-up,
  - segments rotated image,
  - extracts polygons + triangulation + Ekeland features,
  - exports rotated UAV CSV.

### Matching and Visualization

- `FeatureExtractor` (`matching/features.py`): converts each triangle row into feature vectors (interior angles + Ekeland angles, optional shape terms).
- `DistanceMetric` (`matching/features.py`): configurable distance scoring (e.g. `l1`, `l2`, `complex`, `geodesic`, `von_mises`).
- `Matcher` (`matching/features.py`): brute-force matching between all UAV and satellite triangle features; returns top-k results.
- `MatchVisualizer` (`matching/features.py`): overlays matched triangles and rank markers on UAV and satellite images.
- `load_satellite_bounds` (`matching/features.py`): loads geo bounds for mapping GT latitude/longitude to satellite pixels.

### Orchestration Class

- `LocalizationPipeline` (`matching/pipeline.py`): OOP end-to-end runner that chains UAV preprocessing, feature extraction, and matching. In `run.ipynb`, it is instantiated as a pipeline facade for full workflow usage.

## Notes

- The current implementation is notebook-first; `run.ipynb` is the canonical execution path.
- For consistent angle geometry, UAV features should be extracted from the north-aligned (rotated) UAV image.
