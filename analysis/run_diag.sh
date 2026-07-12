#!/bin/bash
cd ~/LocalizationUAV
PY=.venv/bin/python
SITES="01 02 03 04 11"
# wait for build+query chain to finish (or fail)
while ! grep -q "ALL DONE" logs/multisite.log 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] build+query finished, starting diagnostics" | tee logs/diag_status.log
# multi-site truth-anchored retrieval + coverage + Hough (5-D angles, K=20)
echo "[$(date '+%F %T')] phase_a_diag ..." | tee -a logs/diag_status.log
$PY phase_a_diag.py --sites $SITES --limit 100 > logs/diag_multisite.log 2>&1
echo "[$(date '+%F %T')] phase_a_diag rc=$?" | tee -a logs/diag_status.log
# per-site sequential PF (diagnostic + particle filter)
: > logs/pf_multisite.log
for s in $SITES; do
  echo "===== PF site $s =====" >> logs/pf_multisite.log
  $PY pf_pipeline.py --site $s --limit 100 >> logs/pf_multisite.log 2>&1
  echo "[$(date '+%F %T')] PF site $s rc=$?" | tee -a logs/diag_status.log
done
echo "[$(date '+%F %T')] ALL DIAG DONE" | tee -a logs/diag_status.log
