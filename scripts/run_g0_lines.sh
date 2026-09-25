#!/usr/bin/env bash
# G0 for the TRAINING-FREE frontend (LSD line structure): same sites, queries,
# priors and protocol as run_g0.sh; separate cache so nothing is shared with Mask R-CNN.
set -euo pipefail
cd /workspace/KhangTa/LocalizationUAV
PY=../venv/bin/python
while [ ! -f ../data/.visloc_extracted ]; do sleep 30; done
ROOT=$(dirname "$(find ../data/visloc_raw -maxdepth 4 -type d -name 01 -not -path "*__MACOSX*" | head -1)")
ARGS="--root $ROOT --cache cache/visloc_lines --gsd 0.6 --frontend lines"
mkdir -p cache/visloc_lines
$PY -u eval_reg.py satcache $ARGS --sites 05 01 11
[ -f cache/visloc_lines/calib.json ] || $PY -u eval_reg.py calib $ARGS --calib-sites 05 --calib-n 30
cat cache/visloc_lines/calib.json
$PY -u eval_reg.py uavcache $ARGS --sites 01 11 --n 100
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --out results/reg/g0L_uav.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --oracle --out results/reg/g0L_oracle.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --local-radius 50 --out results/reg/g0L_local_uav.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --local-radius 50 --oracle --out results/reg/g0L_local_oracle.csv
$PY analysis/g0_report.py results/reg/g0L_uav.csv results/reg/g0L_oracle.csv results/reg/g0L_local_uav.csv results/reg/g0L_local_oracle.csv
echo G0L_DONE
