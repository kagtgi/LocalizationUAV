#!/bin/bash
cd ~/LocalizationUAV
PY=.venv/bin/python
log(){ echo "[$(date '+%F %T')] $*"; }
# 1. wait until site 03 DB is safely cached, so killing the old job won't lose it
log "waiting for site 03 satellite DB before skipping site 04 ..."
while [ ! -f outputs/eval/03/satellite_kdtree.npz ]; do sleep 45; done
log "site 03 DB present -> stopping original job to skip site 04"
pkill -f run_multisite.sh 2>/dev/null; pkill -f run_diag.sh 2>/dev/null
pkill -f "eval.py --stage build --sites 02 03 04 11" 2>/dev/null
sleep 6
# 2. build ONLY site 11 (01,02,03 cached -> auto-skipped without --force)
log "BUILD site 11 ..."
$PY eval.py --stage build --sites 11 2>&1
log "BUILD 11 done rc=$?"
# 3. query the three new sites (01 already has records)
log "QUERY sites 02 03 11 (100 imgs each) ..."
$PY eval.py --stage query --sites 02 03 11 --limit-images 100 --examples 2 2>&1
log "QUERY done rc=$?"
# 4. diagnostics across all 4 final sites
SITES="01 02 03 11"
log "phase_a_diag ..."
$PY phase_a_diag.py --sites $SITES --limit 100 > logs/diag_multisite.log 2>&1
log "phase_a_diag rc=$?"
: > logs/pf_multisite.log
for s in $SITES; do echo "===== PF site $s =====" >> logs/pf_multisite.log; $PY pf_pipeline.py --site $s --limit 100 >> logs/pf_multisite.log 2>&1; log "PF $s rc=$?"; done
# 5. collect metrics + cross-site figure
log "collect + figure ..."
$PY multisite_collect.py > logs/collect.log 2>&1; log "collect rc=$?"
$PY make_figs3.py >> logs/collect.log 2>&1; log "figure rc=$?"
log "ALL DIAG DONE" | tee logs/final_status.log
