#!/bin/bash
cd ~/LocalizationUAV
PY=.venv/bin/python
echo "[04:37:33] BUILD site 09"
 eval.py --stage build --sites 09 2>&1
echo "[04:37:33] BUILD 09 done rc=0"
 eval.py --stage query --sites 09 --limit-images 100 --examples 2 2>&1
echo "[04:37:33] QUERY 09 done rc=0"
echo SITE09_ALL_DONE
