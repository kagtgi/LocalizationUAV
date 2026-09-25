#!/usr/bin/env bash
# M0 (conference method) on EXACTLY the G0 queries and priors: global + 1 km window.
set -euo pipefail
cd /workspace/KhangTa/LocalizationUAV
PY=../venv/bin/python
while [ ! -f cache/visloc/uav/11/queries.txt ]; do sleep 60; done
ROOT=$(dirname "$(find ../data/visloc_raw -maxdepth 4 -type d -name 01 | head -1)")
for s in 01 11; do
  $PY -u eval.py --stage build --data-root "$ROOT" --model best_model.pth --out outputs/m0 --sites $s
  $PY -u eval.py --stage query --data-root "$ROOT" --model best_model.pth --out outputs/m0 --sites $s \
      --image-list cache/visloc/uav/$s/queries.txt --prior-radius 1000 --prior-seed 0 --examples 0
done
echo M0WIN_DONE
