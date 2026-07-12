#!/bin/bash
set -o pipefail
cd ~/LocalizationUAV
PY=.venv/bin/python
SITES="02 03 04 11"
echo "[$(date '+%F %T')] START multisite build+query sites: $SITES"
echo "[$(date '+%F %T')] BUILD stage ..."
$PY eval.py --stage build --sites $SITES 2>&1
echo "[$(date '+%F %T')] BUILD done (rc=$?)"
echo "[$(date '+%F %T')] QUERY stage (100 imgs/site) ..."
$PY eval.py --stage query --sites $SITES --limit-images 100 --examples 2 2>&1
echo "[$(date '+%F %T')] QUERY done (rc=$?)"
echo "[$(date '+%F %T')] ALL DONE"
