#!/bin/bash
cd ~/LocalizationUAV
PY=.venv/bin/python
: > logs/dmax_retrieval.log
: > logs/dmax_area.log
for d in 0 1 2 4 8; do
  echo "===== max_depth=$d: truth-anchored retrieval (sites 01 02 03) =====" >> logs/dmax_retrieval.log
  $PY phase_a_diag.py --sites 01 02 03 --limit 100 --max-depth $d >> logs/dmax_retrieval.log 2>&1
  echo "[$(date +%T)] retrieval max_depth=$d done rc=$?"
done
for d in 0 1 2 4 8; do
  echo "===== max_depth=$d: area-level recall (sites 01 02 03 08 11) =====" >> logs/dmax_area.log
  $PY area_ekeland_probe2.py --sites 01 02 03 08 11 --limit 100 --sliding 0 --max-depth $d >> logs/dmax_area.log 2>&1
  mv outputs/paper/area_ekeland.json outputs/paper/area_ekeland_d${d}.json 2>/dev/null
  echo "[$(date +%T)] area max_depth=$d done rc=$?"
done
echo DMAX_SWEEP_ALL_DONE
