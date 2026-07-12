import json
from pathlib import Path
for s in ["01","02","03","08","11"]:
    d = Path("outputs/eval")/s
    a = json.loads((d/"area_per_frame.json").read_text()) if (d/"area_per_frame.json").exists() else {}
    b = json.loads((d/"baselines_per_frame.json").read_text()) if (d/"baselines_per_frame.json").exists() else {}
    merged = {**a, **b}
    (d/"merged_per_frame.json").write_text(json.dumps(merged))
    print(s, list(merged.keys()))
