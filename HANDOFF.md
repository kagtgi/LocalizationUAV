# Handoff — journal extension of "Structure-Based UAV Geo-Localization" (as of 2026-09-26 ~04:25 UTC)

## Goal
Q1-journal extension (ISPRS JPRS / TGRS) of the conference paper, per the user's plan
`C:\Users\Win\Downloads\plan_final_a100.md` and the approved plan
`C:\Users\Win\.claude\plans\c-users-win-downloads-plan-final-a100-m-logical-bunny.md`.
The user wants a **geometry-based, training-free** method framed as **mathematical optimization**. It must stay
based on the old **Ekeland free-cone angle**. The paper uses 3 datasets (UAV-VisLoc main, World-UAV, AnyVisLoc) and
full paper material (13 figs, auto-generated tables, supplement) in `paper_journal/`. **Never fabricate numbers.**

## Current method (branch `structreg`, all pushed to GitHub)
"Training-free edge-constrained structural registration":
1. Canonicalize the UAV frame: rotate north-up by IMU yaw; GSD = c·AGL/W_px. AGL = metadata height − DEM median in the
   prior window. c = 2·tan(HFOV/2) is calibrated per camera on a non-test site.
2. Extract oriented line structure with LSD (`localization/registration/lines.py`): segments ≥ 6 m, rasterized with
   K = 8 orientation bins. Persistence weights come from a structural ensemble (image scale × blur).
3. Build the map side as oriented smoothed-chamfer kernels G_k (soft ±1 bin).
4. Search (`fft_search.py`): masked, orientation-stacked NCC, evaluated exactly by FFT for every translation in the
   window per (θ, s). Each (θ, s) score map is z-scored (CFAR) to remove the small-template bias. Templates are built
   lazily per batch.
5. Verify (`linegraph.py`): structural line filtering, junctions (L/T/X), PSLG CDT, **Ekeland free-cone angle at
   junctions**, topological consistency, then RANSAC Sim(2) on the junction pairs.
6. Refine and score (`refine.py`, `integrity.py`): continuous refinement (trilinear over orientation), Laplace
   covariance (finite-difference Hessian), logistic integrity/abstention model.

Segmentation (Mask R-CNN) was **dropped** because it is weak: it misses roofs and marks roads as buildings. The old
code paths remain behind `--frontend maskrcnn`.

**Main runner:** `eval_reg.py`
- Stages: `satcache`, `satgraph`, `calib`, `uavcache`, `run`.
- Key flags: `--frontend lines`, `--radius`, `--local-radius`, `--oracle`, `--verify-graph`, `--isotropic`, `--no-ekeland`, `--agl dem`.
- Priors are paired via crc32 seeds (`prior_uv`).

**Other runners:**
- `eval_wuav.py` (World-UAV Rot) and `eval_avl.py` (AnyVisLoc, OpenGL pose convention verified).
- `eval.py`: M0 baseline with `--image-list` and `--prior-radius`.

**Tests:** `scripts/test_structreg.py` and `scripts/test_oriented.py`. Both pass: 0.1 m error, and graph verification
scores the true pose 0.82 vs 0.001 for a wrong one.

**Reports:** `analysis/g0_report.py` and `analysis/make_figs_reg.py`.

## Compute (A100 via JupyterLab, see memory `a100-jupyter-access`)
- **Access:** https://ws.gvlab.org/fablab/thanhan/lab. The user logs in; never type the password. Commands run through
  a Jupyter kernel websocket. The helpers live in browser `localStorage['claude_helpers']`; re-create them with
  `await eval(localStorage.getItem('claude_helpers'))`, which gives `window.sh`, `window.bg`, `window.G` (git with
  `GIT_CONFIG_GLOBAL`).
- **Layout:** repo `/workspace/KhangTa/LocalizationUAV` (branch `structreg`), venv `/workspace/KhangTa/venv`,
  data `/workspace/KhangTa/data/{visloc_raw/UAV_VisLoc_dataset, worlduav/Rot, anyvisloc/Scene_10..24(partly), dem}`,
  logs `/workspace/KhangTa/logs/`.
- **Limits:** the container has a **20 GB RAM cgroup cap**. The GPU is shared (~10 GB free). Run big-map jobs one at a
  time.
- **Code sync:** commit and push locally, then `git pull` on the server.

## Status right now
- **G0 pipeline** `scripts/run_g0_lines.sh` is running on the server (log `logs/g0L.log`).
  - **Done:** oriented line caches for sites 02/01/11 and map graphs for 01/11.
  - **In progress:** calibration of the 3976-px camera FOV on **site 02**. Localized frames (peak ≤ 25 m from GT in a
    300 m window) give s ≈ **0.54–0.62** vs c0 = 1.8, i.e. HFOV ≈ 50°. About 60% of frames localize, some to 1–15 m.
    This is the first real-data positive signal.
  - **Next, automatically:** `uavcache` for 01 and 11, then local sanity 50 m (uav + oracle), then main 1 km window
    with `--verify-graph`, then oracle, then `g0_report.py` (writes `results/reg/g0_report.md`), then the ablations
    (isotropic, no-verify, no-Ekeland).
- **M0 baseline** `scripts/run_m0win.sh` waits for `G0L_DONE` in `g0L.log`, then builds the M0 database for 01/11 and
  queries the same images and priors.
- **World-UAV:** fails (median ~500–740 m) because the reference segmentation failed. It has not been re-run with the
  lines frontend yet.
- **AnyVisLoc:** convention fixed (OpenGL), but it needs its own scale calibration (s ≈ 0.7–0.85 observed). Lines
  frontend not run yet. Maps are tiny (~225 m) and ~45° oblique, so it is a validity-envelope experiment.

## G0 decision rule (locked by the user)
- **Metrics:** Median / Mean / P95 error, S@25/50/100, candidate recall, stratified by query structure count.
- **Clearly beats random + M0 on both sites:** go. UAV-VisLoc becomes the main result; World-UAV and AnyVisLoc become
  validity-envelope experiments.
- **Only one site beats them:** analyse structural observability.
- **Both fail but oracle is good:** the bottleneck is extraction.
- **Both fail and oracle also fails:** stop the Sim(2) direction and redesign; do not write the Q1 story.

## Key data pitfalls found (memory `visloc-metadata-pitfalls`)
- UAV-VisLoc `height` is **above sea level**: ground ≈ 10 m (01), ≈ 1,941 m (05), ≈ 1,783 m (11). Use the DEM.
- Cameras differ per site: 3976 px at 01–04 and 06, 3000 px at 05.
- EXIF is stripped from the images.
- `02.csv` has a header row.
- NCC is biased toward small templates. z-scoring the score maps mitigates this but does not fully fix it; frames that
  snap to s = 0.25 are failures.

## Paper (`paper_journal/`)
- **Written:** `main.tex`, and sections `intro`, `related`, `method` (the lines method), `algorithms`, `experiments`
  (setup + why-M0 only), `appendix` (proofs of Props 1–3), `conclusion` (placeholder).
- **Tables:** `tables/T0_notation`.
- **Bibliography:** `references.bib`; all new entries were verified.
- **Rule:** abstract, results and conclusion must wait for G0 numbers. Figures use the `paper-schematic-figures`,
  `nature-figure-style` and `dataviz` skills.

## Open TODOs
1. Finish G0, read `results/reg/g0_report.md`, and apply the decision rule. Report honestly to the user (in Vietnamese).
2. If go: integrity fitting (leave-one-site-out), all 11 sites, World-UAV and AnyVisLoc with the lines frontend,
   figures and tables, then write results.
3. Revisit M0 in light of the height bug (ASL, not AGL): it may have contributed to M0's failure.
4. **When the work is complete, make the GitHub repo private again:**
   `gh repo edit kagtgi/LocalizationUAV --visibility private --accept-visibility-change-consequences`.
5. Pip/OpenCV note: never install into the system Python on the shared server; use the venv.
