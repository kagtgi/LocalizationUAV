#!/bin/bash
cd ~/LocalizationUAV
PY=.venv/bin/python
: > logs/pf_multisite2.log
# NOTE: site 08 excluded here - its records_main.csv now holds EVENLY-SAMPLED
# frames (fix for coarse-localization coverage), which breaks PF's consecutive-
# trajectory assumption. Site 08's PF trajectory is queried separately below
# with a genuine consecutive window.
for s in 01 02 03 11; do
  echo "===== PF site $s (trajectory 1: frames 1-100) =====" >> logs/pf_multisite2.log
  $PY pf_pipeline.py --site $s --limit 100 >> logs/pf_multisite2.log 2>&1
  echo "[$(date +%T)] PF site $s done rc=$?"
done
echo PF_MULTISITE_ALL_DONE
