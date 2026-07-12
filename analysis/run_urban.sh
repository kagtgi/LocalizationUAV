#!/bin/bash
cd ~/LocalizationUAV
PY=.venv/bin/python
log(){ echo "[$(date '+%F %T')] $*"; }
log "BUILD sites 08 09 ..."
$PY eval.py --stage build --sites 08 09 2>&1
log "BUILD 08 09 done rc=$?"
log "QUERY sites 08 09 (100 imgs each) ..."
$PY eval.py --stage query --sites 08 09 --limit-images 100 --examples 2 2>&1
log "QUERY 08 09 done rc=$?"
SITES="01 02 03 08 09 11"
log "phase_a_diag over all 6 ..."
$PY phase_a_diag.py --sites $SITES --limit 100 > logs/diag_multisite.log 2>&1
log "phase_a_diag rc=$?"
: > logs/pf_multisite.log
for s in $SITES; do echo "===== PF site $s =====" >> logs/pf_multisite.log; $PY pf_pipeline.py --site $s --limit 100 >> logs/pf_multisite.log 2>&1; log "PF $s rc=$?"; done
log "collect + figure ..."
$PY multisite_collect.py > logs/collect.log 2>&1; log "collect rc=$?"
$PY make_figs3.py >> logs/collect.log 2>&1; log "figure rc=$?"
log "ALL URBAN DONE" | tee logs/urban_status.log
