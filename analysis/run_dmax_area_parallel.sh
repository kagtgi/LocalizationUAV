#!/bin/bash
cd ~/LocalizationUAV
PY=.venv/bin/python
: > logs/dmax_area_parallel.log
for d in 0 1 2 8; do
  echo "===== max_depth=$d: area-level recall (sites 01 02 03 08 11) =====" >> logs/dmax_area_parallel.log
  $PY area_ekeland_probe2.py --sites 01 02 03 08 11 --limit 100 --sliding 0 --max-depth $d >> logs/dmax_area_parallel.log 2>&1
  mv outputs/paper/area_ekeland.json outputs/paper/area_ekeland_d${d}.json 2>/dev/null
  echo "[$(date +%T)] area max_depth=$d done rc=$?"
done
echo DMAX_AREA_PARALLEL_DONE
