# Legacy code

This folder contains code that is **not** part of the paper's pipeline.
Everything in here is preserved for reference, ablation, or historical
context, but the production pipeline in `localization/` does not import it.

| File | What it is | Why it's here |
|------|------------|---------------|
| `mutual_learning_swin.py` | A Swin-Transformer + ResNet-50 dual-Mask R-CNN mutual-learning experiment | Unrelated experimental track from before the geometric pipeline was finalized. |
| `brute_force_matcher.py` | Original `O(M*N)` triangle-pair matcher with optional shape-feature flags | Replaced by `localization/database/kdtree.py` + `localization/matching/query.py`, which use `scipy.spatial.KDTree(p=1)` and plurality voting per the paper. |
| `extra_distance_metrics.py` | `complex`, `geodesic`, `circular_mean`, `von_mises` distance variants | Paper §4.5 specifies `ell_1` (cityblock) only. These remain for reference and possible ablation studies. |

To re-run anything in `legacy/`, import directly:

```python
from legacy.brute_force_matcher import legacy_brute_force_matching
```
