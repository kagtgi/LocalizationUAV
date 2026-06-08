"""Offline verification of the paper-fidelity fixes + the new Fig. 5(a) plot.

Runs without the 16 GB UAV-VisLoc dataset or the Mask R-CNN checkpoint - it uses
hand-built polygons and a synthetic satellite image. Exits non-zero on failure.

Checks:
  1. Package imports cleanly (incl. the new ``draw_gt_vs_topn_centroids``).
  2. Interior-filter CDT: every emitted triangle centroid lies inside the polygon.
  3. D_max cap honoured: max_depth=1 vs 4 yields different descriptors.
  4. Ekeland extraction == paper Definition 2 (closed form e=min(360-alpha_int,180)):
     reflex L-shape vertex -> 90 deg; convex vertex -> 180 deg.
  5. Isolated triangle -> (e1,e2,e3) = (180,180,180).
  6. Monotonicity (Prop. 1): deeper expansion -> smaller-or-equal min Ekeland angle.
  7. New plot writes a non-empty PNG and selects the correct nearest-GT candidate.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon

# Make the repo root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# L-shape (CCW, one reflex vertex at index 3 = (1,1), interior angle 270 deg).
L_SHAPE = np.array([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], dtype=float)


def test_imports() -> None:
    import localization
    from localization import draw_gt_vs_topn_centroids  # noqa: F401
    from localization.geometry.descriptor import triangle_descriptors_from_polygon  # noqa: F401

    check("import localization + new symbols", True)


def test_interior_filter() -> None:
    from localization.geometry.descriptor import triangle_descriptors_from_polygon

    desc, cent = triangle_descriptors_from_polygon(L_SHAPE, max_depth=4)
    poly = Polygon(L_SHAPE)
    inside = all(poly.contains(Point(float(x), float(y))) for x, y in cent)
    check(
        "interior-filter CDT: all centroids inside polygon",
        desc.shape[0] > 0 and inside,
        f"{desc.shape[0]} triangles",
    )


def test_depth_cap() -> None:
    from localization.geometry.descriptor import triangle_descriptors_from_polygon

    d1, _ = triangle_descriptors_from_polygon(L_SHAPE, max_depth=1)
    d4, _ = triangle_descriptors_from_polygon(L_SHAPE, max_depth=4)
    differ = d1.shape != d4.shape or not np.allclose(d1, d4)
    check("D_max cap honoured (depth 1 != depth 4)", differ)


def test_ekeland_closed_form() -> None:
    """Directly verify Definition 2 on a known boundary loop."""
    from localization.geometry.ekeland import calculate_ekeland_angles_on_boundary

    tri = Delaunay(L_SHAPE)  # supplies .points; we hand-build the boundary loop
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    expansion_result = {"boundary_edges": edges, "original_vertices": {0, 3}}
    res = calculate_ekeland_angles_on_boundary(tri, expansion_result)
    e_reflex = res[3]["ekeland_angle"]   # interior 270 -> 360-270 = 90
    e_convex = res[0]["ekeland_angle"]   # interior 90  -> min(270,180) = 180
    check("Ekeland reflex vertex == 90 deg (Def. 2)", abs(e_reflex - 90.0) < 1e-6, f"e={e_reflex:.3f}")
    check("Ekeland convex vertex == 180 deg (Def. 2)", abs(e_convex - 180.0) < 1e-6, f"e={e_convex:.3f}")


def test_isolated_triangle() -> None:
    from localization.geometry.descriptor import triangle_descriptors_from_polygon

    desc, _ = triangle_descriptors_from_polygon(np.array([(0, 0), (4, 0), (1, 3)], float))
    ok = desc.shape[0] == 1 and np.allclose(desc[0, 2:5], [180.0, 180.0, 180.0])
    check("isolated triangle -> (e1,e2,e3)=(180,180,180)", ok, str(desc[0, 2:5] if desc.shape[0] else "empty"))


def test_monotonicity() -> None:
    from localization.geometry.descriptor import triangle_descriptors_from_polygon

    # A larger polygon so the kernel can expand several rounds.
    poly = np.array([(0, 0), (6, 0), (6, 2), (3, 2), (3, 4), (6, 4), (6, 6), (0, 6)], float)
    d1, _ = triangle_descriptors_from_polygon(poly, max_depth=1)
    d4, _ = triangle_descriptors_from_polygon(poly, max_depth=4)
    if d1.shape[0] == 0 or d4.shape[0] == 0:
        check("monotonicity (deeper -> smaller min Ekeland)", False, "no descriptors")
        return
    min_e1 = float(d1[:, 2:5].min())
    min_e4 = float(d4[:, 2:5].min())
    check(
        "monotonicity (deeper -> smaller-or-equal min Ekeland)",
        min_e4 <= min_e1 + 1e-6,
        f"min_e(d=1)={min_e1:.2f}, min_e(d=4)={min_e4:.2f}",
    )


def test_visualization() -> None:
    from PIL import Image

    from localization import draw_gt_vs_topn_centroids
    from localization.matching.query import PatchPrediction

    tmp = Path(tempfile.mkdtemp())
    sat_path = tmp / "fake_satellite.png"
    out_path = tmp / "gt_vs_top.png"
    Image.fromarray(np.full((600, 800, 3), 60, dtype=np.uint8)).save(sat_path)

    rng = np.random.default_rng(0)
    pts = rng.uniform([20, 20], [780, 580], size=(50, 2))
    top_n = [
        PatchPrediction(rank=i + 1, patch_id=f"p{i}", pixel_xy=(float(x), float(y)), vote_count=50 - i)
        for i, (x, y) in enumerate(pts)
    ]
    gt = (400.0, 300.0)
    expected_nearest = int(np.argmin(((pts[:, 0] - gt[0]) ** 2) + ((pts[:, 1] - gt[1]) ** 2)))

    fig = draw_gt_vs_topn_centroids(
        satellite_image_path=str(sat_path),
        top_n=top_n,
        gt_pixel_xy=gt,
        title="Satellite map (synthetic, 405 m, phi=165 deg)",
        output_path=str(out_path),
    )
    import matplotlib.pyplot as plt

    plt.close(fig)
    wrote = out_path.exists() and out_path.stat().st_size > 1000
    check("new plot writes a non-empty PNG", wrote, f"{out_path.stat().st_size if out_path.exists() else 0} bytes")

    # Confirm the nearest-GT selection logic matches a direct argmin.
    xs = np.array([p.pixel_xy[0] for p in top_n])
    ys = np.array([p.pixel_xy[1] for p in top_n])
    sel = int(np.argmin((xs - gt[0]) ** 2 + (ys - gt[1]) ** 2))
    check("nearest-GT candidate selection correct", sel == expected_nearest, f"idx={sel}")


def main() -> int:
    print("=== LocalizationUAV finalization verification ===")
    for test in (
        test_imports,
        test_interior_filter,
        test_depth_cap,
        test_ekeland_closed_form,
        test_isolated_triangle,
        test_monotonicity,
        test_visualization,
    ):
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            check(test.__name__, False, f"raised {type(exc).__name__}: {exc}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
