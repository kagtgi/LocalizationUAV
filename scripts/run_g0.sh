#!/usr/bin/env bash
# G0 go/no-go gate (locked protocol): UAV-VisLoc sites 01 + 11, 100 queries/site,
# 1 km prior window. Calibration of the camera FOV constant on site 02 only (same camera as 01/11, not a G0 site).
set -euo pipefail
cd /workspace/KhangTa/LocalizationUAV
PY=../venv/bin/python
D=../data

# 1) wait for the download (and any unzip already in flight), then extract once
while pgrep -f "[w]get.*uavvisloc.zip" > /dev/null; do sleep 30; done
while pgrep -f "[u]nzip -q -o ../data/uavvisloc.zip" > /dev/null; do sleep 20; done
if [ ! -f $D/.visloc_extracted ] && [ -d $D/visloc_raw/UAV_VisLoc_dataset/11 ]; then touch $D/.visloc_extracted; fi
if [ ! -f $D/.visloc_extracted ]; then
  mkdir -p $D/visloc_raw && unzip -q -o $D/uavvisloc.zip -d $D/visloc_raw && touch $D/.visloc_extracted
fi
# locate the folder that contains 01/ and the bounds csv
ROOT=$(dirname "$(find $D/visloc_raw -maxdepth 4 -type d -name 01 -not -path "*__MACOSX*" | head -1)")
echo "UAV-VisLoc root: $ROOT"; ls "$ROOT"
ARGS="--root $ROOT --cache cache/visloc --gsd 0.6"

# 2) satellite caches (G0 sites + calibration site)
$PY -u eval_reg.py satcache $ARGS --sites 05 01 11
# 3) calibrate GSD factor on site 05 only (never on G0 sites)
[ -f cache/visloc/calib.json ] || $PY -u eval_reg.py calib $ARGS --calib-sites 05 --calib-n 30
cat cache/visloc/calib.json
# 4) query caches
$PY -u eval_reg.py uavcache $ARGS --sites 01 11 --n 100
# 5) runs (same queries, same priors via stable seeds)
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --out results/reg/g0_uav.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --radius 1000 --oracle --out results/reg/g0_oracle.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --local-radius 50 --out results/reg/g0_local_uav.csv
$PY -u eval_reg.py run $ARGS --sites 01 11 --n 100 --local-radius 50 --oracle --out results/reg/g0_local_oracle.csv
$PY analysis/g0_report.py results/reg/g0_uav.csv results/reg/g0_oracle.csv results/reg/g0_local_uav.csv results/reg/g0_local_oracle.csv
echo G0_DONE
