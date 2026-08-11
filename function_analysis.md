# Function Analysis

This document summarizes the Python files in the repository **based only on docstrings / description notes**, without reading implementation details.

It excludes the duplicated copy under `notebooks/LocalizationUAV/...` and focuses on the main repository.

---

# 1. `localization/`

## `localization/__init__.py`
- Top-level public API of the package.
- Uses **lazy imports** so torch-heavy modules are only imported when needed.
- Goal: geometry / KD-tree code can be used without paying torch import cost.

## `localization/config.py`
- Contains **runtime config dataclasses** aligned with the paper.
- Main groups:
  - `PathConfig`: paths for one UAV-VisLoc flight.
  - `IndexConfig`: patching + KDTree hyperparameters.
  - `TrainConfig`: Mask R-CNN training hyperparameters.

## `localization/utils.py`
- Generic helpers for:
  - device selection,
  - reproducibility seeding,
  - loading JSON with multiple encodings,
  - saving training checkpoints.

---

# 2. `localization/database/`

## `localization/database/__init__.py`
- Entry point for the database layer.
- Describes the **satellite descriptor database** portion: patching + KDTree.
- Uses lazy imports so `kdtree` stays torch-free.

## `localization/database/patches.py`
- Splits a large satellite image into **500×500 patches with stride 100**.
- Main functions:
  - `patch_count()`: counts total patches.
  - `iter_patches()`: streams patches with `patch_id` and top-left coordinates.

## `localization/database/builder.py`
- Builds **per-patch satellite descriptors**.
- Uses the same `segment_batch` path for both satellite and UAV images to keep segmentation behavior consistent.
- Main functions:
  - `_polygons_to_descriptors()`: polygon → 5-D descriptors + centroids.
  - `extract_patch_descriptors()`: process **one image** into descriptors + centroids (used for UAV query).
  - `build_satellite_descriptors()`: scan a full GeoTIFF and return a DataFrame of descriptors.
- According to the current docstring:
  - supports `checkpoint_dir`,
  - writes CSV chunks + `manifest.json`,
  - resumes after crashes,
  - **does not save** the KD-tree `.npz`; that belongs to `kdtree.py`.

## `localization/database/kdtree.py`
- Wrapper for the **KD-tree database** of satellite descriptors.
- Uses `scipy.spatial.KDTree` with `p=1` (`ell_1`, cityblock), matching the paper.
- Main class: `SatelliteDatabase`
  - stores descriptors, centroids, and patch identifiers,
  - builds the KDTree lazily on first query,
  - supports KNN query,
  - supports save/load to `.npz`.
- Documented methods/properties include:
  - `patch_ids`, `patch_id_codes`, `patch_id_strings`,
  - `patch_centroids`,
  - `from_dataframe()`,
  - `query()`,
  - `save()`.

---

# 3. `localization/geometry/`

## `localization/geometry/__init__.py`
- No notable docstring.

## `localization/geometry/descriptor.py`
- Defines the **paper-locked 5-D descriptor**:
  - `f = (alpha_1, alpha_2, e_1, e_2, e_3)`.
- Explicitly states that extra features outside the paper specification are not part of the intended production descriptor.
- Main functions:
  - `interior_angles()`: compute the three triangle interior angles.
  - `triangle_descriptor()`: assemble the 5-D feature.
  - `triangle_descriptors_from_polygon()`: polygon → triangulation → descriptors + centroids.

## `localization/geometry/ekeland.py`
- Computes **Ekeland / MFCA angles** for triangle vertices.
- Uses the kernel polygon `P_sub` produced by triangulation expansion.
- Docstring says the axis-parallel reflection construction is reduced to a closed-form O(1) formula based on the interior angle of `P_sub`.
- Main items:
  - `_outward_bisector()`
  - `EkelandAnalyzer`: computes Ekeland angles across a triangulation.

## `localization/geometry/triangulation.py`
- Handles **CDT + Phase 1 kernel expansion**.
- Implements the idea of merging neighboring triangles into `P_sub` while keeping the three seed vertices on the boundary.
- Main class:
  - `TriangulationGraph`
    - builds the dual graph,
    - orders polygon boundaries,
    - expands from a seed triangle under Ekeland criticality conditions.

## `localization/geometry/visualize.py`
- Standalone visualization for:
  - one CDT triangle expansion,
  - Ekeland angle geometry.
- Intended for sanity-checking the geometry pipeline.

---

# 4. `localization/io/`

## `localization/io/__init__.py`
- No notable docstring.

## `localization/io/bounds.py`
- Converts between:
  - satellite bounds,
  - latitude/longitude,
  - pixels,
  - meter offsets.
- Main functions:
  - `_default_bounds_csv_candidates()`
  - `load_satellite_bounds()`
  - `latlon_to_pixel()`
  - `pixel_to_latlon()`
  - `estimate_satellite_resolution_meters()`
  - `pixel_offset_to_meters()`

## `localization/io/dataset.py`
- Helpers for the **UAV-VisLoc dataset**.
- Main class:
  - `VisLocFlight`: abstraction for one flight (`01`, `02`, ...).
- Main functions:
  - `load_flight_metadata()`: load per-image GPS + heading metadata.
  - `get_image_pose()`: look up metadata row by drone image filename.

## `localization/io/export.py`
- Exports expansion / Ekeland results to CSV.
- Main class:
  - `FeatureCsvExporter`: writes a uniform CSV schema.

---

# 5. `localization/matching/`

## `localization/matching/__init__.py`
- Entry point for **query / vote / top-N / visualization**.
- `visualize` is lazy-loaded because it depends on OpenCV + matplotlib.

## `localization/matching/query.py`
- Implements the **online query** stage:
  - KNN retrieval under `ell_1`,
  - plurality voting over `patch_id`,
  - rank-1 patch prediction.
- According to the docstring:
  - `query_uav()` returns the headline output of the paper.
  - `query_top_n_patches()` supports analysis of top-N candidates.
- Main items:
  - `PatchPrediction`: one ranked candidate patch.
  - `plurality_vote()`
  - `_plurality_vote_codes()`
  - `_top_n_from_counts()`
  - `query_uav()`
  - `query_top_n_patches()`

## `localization/matching/visualize.py`
- Visualization for query results.
- Main figure types:
  - rank-1 prediction + GT,
  - top-N prediction overlays.
- Main functions:
  - `draw_gt_and_prediction()`
  - `draw_top_n_predictions()`
  - `render_localization_result()`
  - `render_top_n_result()`
- Also contains legacy wrappers:
  - `draw_predicted_position`
  - `render_match_figure`

---

# 6. `localization/preprocess/`

## `localization/preprocess/__init__.py`
- No notable docstring.

## `localization/preprocess/uav.py`
- Implements **UAV preprocessing** from paper §4.1.
- Responsibilities:
  - scale normalization using altitude,
  - yaw alignment,
  - roll/pitch correction,
  - center crop,
  - resize to 500×500.
- Main items:
  - `UAVPreprocessor`
  - `process_uav()` functional API

---

# 7. `localization/segmentation/`

## `localization/segmentation/__init__.py`
- Entry point for building segmentation.
- Uses lazy imports for torch / OpenCV / scikit-image so geometry code can stay lightweight.

## `localization/segmentation/contours.py`
- Utilities to extract polygons from masks.
- Focuses on Douglas-Peucker simplification and outlier filtering.
- Main functions:
  - `extract_contours_from_mask()`
  - `contour_to_polygon()`
  - `contour_to_polygon_dynamic()`
  - `filter_large_polygons()`
  - `filter_large_polygons_dynamic()`

## `localization/segmentation/inference.py`
- **Mask R-CNN inference + polygon extraction**.
- Emphasizes that both UAV and satellite paths use the same `segment_batch()` function at the same 500×500 input size.
- Main functions:
  - `_postprocess_one()`
  - `segment_batch()`
  - `segment_image()`
  - `_pyramid_blend_weight()`
  - `segment_image_patchwise()`
  - `polygons_from_soft_mask()`
- Docstring notes:
  - `segment_image_patchwise()` is only for full-size satellite mask use cases,
  - the standard paper pipeline does **not** use that path.

## `localization/segmentation/model.py`
- Defines the learned segmentation component.
- Provides:
  - Mask R-CNN ResNet-50-FPN construction,
  - warmup LR scheduler,
  - checkpoint loading to an eval-ready model.
- Main functions:
  - `get_model()`
  - `warmup_lr_scheduler()`
  - `load_model()`

---

# 8. `scripts/`

## `scripts/train_maskrcnn.py`
- No single concise module docstring, but the parsed docstrings describe the training utilities:
  - `ModelEMA`: exponential moving average shadow model.
  - `build_optimizer()`: layer-wise learning-rate strategy for Mask R-CNN.
  - `build_scheduler()`: batch/epoch scheduler selection.
  - `evaluate_loss()`: evaluate Mask R-CNN loss.
  - `visualize_predictions()`: visualize predictions vs GT over epochs.
- Overall this is the **Mask R-CNN training script**, with features like DDP / AMP / EMA / scheduler support.

## `scripts/test_maskrcnn.py`
- Few docstrings were present in the parsed output.
- Based on the file header comments, it is the **Mask R-CNN testing script** for GF-7 data using a checkpoint `.pth`, with options to save visualizations and raw masks.

---

# 9. `legacy/`

## `legacy/brute_force_matcher.py`
- Old brute-force matcher with `O(M*N)` complexity.
- Explicitly documented as **not used** in the production pipeline.
- Preserved as reference from earlier code.
- Main functions:
  - `legacy_extract_features_from_row()`
  - `legacy_brute_force_matching()`

## `legacy/extra_distance_metrics.py`
- Extra distance metrics **outside the paper specification**.
- The docstring explicitly says the paper uses only `ell_1`; these are exploratory / ablation / reference metrics.
- Main functions:
  - `angular_distance()`
  - `complex_angle_distance()`
  - `geodesic_feature_distance()`
  - `circular_mean_distance()`
  - `von_mises_feature_score()`

## `legacy/mutual_learning_swin.py`
- Experimental script for **mutual deep learning** between:
  - Mask R-CNN ResNet50,
  - Mask R-CNN Swin-T.
- Main classes mentioned in docstrings:
  - `SwinTransformerBackbone`
  - `MaskRCNNFPNHook`
- This is an experiment branch, not the main paper pipeline.

---

# Big-Picture Summary by Folder

## Core pipeline folders
- `preprocess/` → normalize UAV image
- `segmentation/` → building masks → polygons
- `geometry/` → CDT + Ekeland + descriptor 5-D
- `database/` → build satellite descriptor DB + KDTree
- `matching/` → query + vote + visualize
- `io/` → dataset, bounds, and error conversion helpers

## Supporting / non-main-pipeline folders
- `scripts/` → train/test the segmentation model
- `legacy/` → archived code, ablations, and experiments not used in the main paper pipeline
