#!/usr/bin/env bash
# G0 for the TRAINING-FREE edge-based method: oriented line structure (K orientation
# channels), masked NCC over Sim(2) by FFT, line-graph (junction/CDT/Ekeland) verification.
# Same sites, queries, priors and protocol as the locked G0. Runs sequentially (20 GB cap).
set -euo pipefail
cd /workspace/KhangTa/LocalizationUAV
PY=../venv/bin/python
while [ ! -f ../data/.visloc_extracted ]; do sleep 30; done
ROOT=$(dirname "$(find ../data/visloc_raw -maxdepth 4 -type d -name 01 -not -path "*__MACOSX*" | head -1)")
ARGS="--root $ROOT --cache cache/visloc_lines --gsd 0.6 --frontend lines"
R=results/reg
mkdir -p cache/visloc_lines $R
$PY -u eval_reg.py satcache $ARGS --sites 05 01 11
$PY -u eval_reg.py satgraph $ARGS --sites 01 11
[ -f cache/visloc_lines/calib.json ] || $PY -u eval_reg.py calib $ARGS --calib-sites 05 --calib-n 30
cat cache/visloc_lines/calib.json
$PY -u eval_reg.py uavcache $ARGS --sites 01 11 --n 100
# (1) sanity: does the objective peak at the right place at all? (50 m around GT)
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --local-radius 50 --out $R/g0L_local_uav.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --local-radius 50 --oracle --out $R/g0L_local_oracle.csv
# (2) main: 1 km prior window, oriented + graph verification (+Ekeland)
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --verify-graph --out $R/g0L_uav.csv
# (3) oracle structure, same setting
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --oracle --out $R/g0L_oracle.csv
$PY analysis/g0_report.py $R/g0L_uav.csv $R/g0L_oracle.csv $R/g0L_local_uav.csv $R/g0L_local_oracle.csv
echo G0L_DONE
# (4) ablations (after the gate report)
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --isotropic --out $R/g0L_abl_isotropic.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --out $R/g0L_abl_noverify.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --verify-graph --no-ekeland --out $R/g0L_abl_noekeland.csv
echo G0L_ABL_DONE
