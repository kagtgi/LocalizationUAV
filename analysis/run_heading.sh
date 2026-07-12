#!/bin/bash
cd ~/LocalizationUAV
PY=.venv/bin/python
for deg in 2 5 10; do
  for site in 01 03; do
    echo "[$(date +%T)] heading_noise=$deg site=$site"
    $PY eval.py --stage query --sites $site --variant heading_noise --heading-noise $deg --limit-images 100 2>&1
  done
done
echo HEADING_ALL_DONE
