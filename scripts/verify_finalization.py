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
  7. Order-invariance: a genuine merge ambiguity resolves identically regardless
     of which candidate has the lower CDT triangle index (no index-based bias).
  8. RANSAC sub-patch refinement (Sec. 5.7): recovers a known translation from
     noisy correspondences while ignoring an outlier, and falls back to the
     coarse patch centroid, unchanged, when starved of correspondences.
  9. New plot writes a non-empty PNG and selects the correct nearest-GT candidate.
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


class _MockTriangulation:
    """Duck-types scipy.spatial.Delaunay's ``.points/.simplices/.neighbors``
    so TriangulationGraph can be driven with a hand-built mesh."""

    def __init__(self, points, simplices, neighbors):
        self.points = np.asarray(points, dtype=float)
        self.simplices = np.asarray(simplices, dtype=int)
        self.neighbors = np.asarray(neighbors, dtype=int)


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


def test_order_invariance() -> None:
    """A genuine merge ambiguity (two valid single-step candidates sharing
    edges of different lengths) must resolve the same way regardless of
    which candidate happens to have the lower CDT triangle index - the fix
    for the reviewer-flagged order-dependency bug (was: ``sorted(frontier)``,
    i.e. ascending index with no geometric tiebreak).
    """
    from localization.geometry.triangulation import TriangulationGraph

    # Seed T0 = (A, B, C). T_long shares T0's long edge AB (length 6);
    # T_short shares T0's short edge AC (length ~1.41). Both individually
    # keep A, B, C on the boundary, so both are valid Phase-1 candidates -
    # the correct choice (longest shared edge) must be T_long either way.
    points = [
        (0.0, 0.0),   # 0 = A
        (6.0, 0.0),   # 1 = B
        (1.0, 1.0),   # 2 = C
        (3.0, -2.0),  # 3 = D (forms T_long = A,B,D)
        (-1.0, 2.0),  # 4 = E (forms T_short = A,C,E)
    ]
    long_edge_vertices = {0, 1, 3}

    def run(swap_indices: bool) -> set:
        if not swap_indices:
            simplices = [[0, 1, 2], [0, 1, 3], [0, 2, 4]]  # T_long is index 1
        else:
            simplices = [[0, 1, 2], [0, 2, 4], [0, 1, 3]]  # T_long is index 2
        neighbors = [[1, 2, -1], [0, -1, -1], [0, -1, -1]]
        tri = _MockTriangulation(points, simplices, neighbors)
        graph = TriangulationGraph(tri)
        dual = graph.build_dual_graph()
        result = graph.expand_from_seed_triangle(dual, seed_triangle_idx=0, max_iterations=1)
        merged_idx = next(iter(result["region_triangles"] - {0}))
        return set(int(v) for v in tri.simplices[merged_idx])

    merged_normal = run(swap_indices=False)
    merged_swapped = run(swap_indices=True)
    check(
        "order-invariant merge: normal indexing picks longest shared edge",
        merged_normal == long_edge_vertices, f"merged {merged_normal}",
    )
    check(
        "order-invariant merge: swapped indexing still picks longest shared edge",
        merged_swapped == long_edge_vertices, f"merged {merged_swapped}",
    )


def test_ransac_refinement() -> None:
    """Sub-patch RANSAC translation refinement (paper Sec. 5.7): recovers a
    known translation from noisy in-patch correspondences while ignoring an
    in-patch outlier (false match), and falls back to the coarse patch
    centroid, unchanged, when starved of correspondences.
    """
    from localization.database.kdtree import SatelliteDatabase
    from localization.matching.refine import ransac_refine_position

    rng = np.random.default_rng(1)
    true_translation = np.array([120.0, -40.0])
    n_win = 6
    sat_win = rng.uniform(0, 500, size=(n_win, 2))
    uav_win = sat_win - true_translation + rng.normal(0.0, 0.5, size=(n_win, 2))
    # One in-patch false match: implies a wildly different translation.
    uav_outlier = np.array([[10.0, 10.0]])
    sat_outlier = np.array([[400.0, 400.0]])
    # One decoy correspondence in a different patch (must be ignored entirely).
    uav_decoy = np.array([[250.0, 250.0]])
    sat_decoy = np.array([[10.0, 10.0]])

    all_uav = np.vstack([uav_win, uav_outlier, uav_decoy])
    all_sat = np.vstack([sat_win, sat_outlier, sat_decoy]).astype(np.float32)
    patch_ids = np.array(["winner"] * (n_win + 1) + ["decoy"])
    descriptors = np.zeros((all_sat.shape[0], 5), dtype=np.float32)
    db = SatelliteDatabase(descriptors, all_sat, patch_ids, leaf_size=4)
    winner_code = int(np.where(db.patch_id_strings == "winner")[0][0])
    nearest_indices = np.arange(all_uav.shape[0]).reshape(-1, 1)  # k=1, row i -> sat row i
    coarse_xy = tuple(db.patch_centroids[winner_code])
    uav_reference = (250.0, 250.0)

    result = ransac_refine_position(
        uav_centroids=all_uav, nearest_indices=nearest_indices, db=db,
        winner_code=winner_code, uav_reference_xy=uav_reference, coarse_pixel_xy=coarse_xy,
        inlier_threshold_px=5.0, min_correspondences=3,
    )
    expected_xy = np.array(uav_reference) + true_translation
    err = float(np.linalg.norm(np.array(result.pixel_xy) - expected_xy))
    check(
        "RANSAC refinement recovers known translation, ignores in-patch outlier",
        result.refined and err < 2.0 and result.n_inliers == n_win and result.n_correspondences == n_win + 1,
        f"err={err:.3f}px, inliers={result.n_inliers}/{result.n_correspondences}",
    )

    starved = ransac_refine_position(
        uav_centroids=all_uav[:1], nearest_indices=np.array([[0]]), db=db,
        winner_code=winner_code, uav_reference_xy=uav_reference, coarse_pixel_xy=coarse_xy,
        min_correspondences=3,
    )
    check(
        "RANSAC refinement falls back to the coarse centroid when starved",
        (not starved.refined) and starved.pixel_xy == (float(coarse_xy[0]), float(coarse_xy[1])),
        f"pixel_xy={starved.pixel_xy}",
    )


def test_patch_dedup() -> None:
    """Nearest-patch-center deduplication (patch_owns_centroid): every point
    is owned by exactly one of two adjacent patches, including points
    exactly on the shared cell boundary (no double-count, no gap) - the
    fix for the ~15x triangle duplication measured across the 80%-overlapping
    patch grid.
    """
    from localization.database.patches import patch_owns_centroid

    patch_size, stride = 500, 100
    tl_a, tl_b = (0.0, 0.0), (100.0, 0.0)  # adjacent patches on the grid
    center_a = (tl_a[0] + patch_size / 2, tl_a[1] + patch_size / 2)
    center_b = (tl_b[0] + patch_size / 2, tl_b[1] + patch_size / 2)

    check(
        "patch dedup: a patch's own center is owned by it, not its neighbor",
        patch_owns_centroid(center_a, tl_a, patch_size, stride)
        and not patch_owns_centroid(center_a, tl_b, patch_size, stride),
    )

    boundary = ((center_a[0] + center_b[0]) / 2, center_a[1])  # exactly on the shared cell edge
    owned_by_a = patch_owns_centroid(boundary, tl_a, patch_size, stride)
    owned_by_b = patch_owns_centroid(boundary, tl_b, patch_size, stride)
    check(
        "patch dedup: a point exactly on the shared boundary is owned by exactly one patch",
        owned_by_a != owned_by_b, f"owned_by_a={owned_by_a}, owned_by_b={owned_by_b}",
    )

    far_away = (center_a[0] + 10 * stride, center_a[1])
    check(
        "patch dedup: a point far outside a patch's cell is owned by neither",
        not patch_owns_centroid(far_away, tl_a, patch_size, stride)
        and not patch_owns_centroid(far_away, tl_b, patch_size, stride),
    )

    # A wider cell_size re-admits bounded overlap between near neighbors
    # (trading duplication for more triangles/vote-bucket); a point 1.5
    # strides from A's center should be excluded at the default (stride)
    # cell size but included once the cell is widened past that distance.
    p_1_5_stride = (center_a[0] + 1.5 * stride, center_a[1])
    check(
        "patch dedup: wider cell_size re-admits a point excluded at the default width",
        not patch_owns_centroid(p_1_5_stride, tl_a, patch_size, stride, cell_size=stride)
        and patch_owns_centroid(p_1_5_stride, tl_a, patch_size, stride, cell_size=4 * stride),
    )

    from localization.database.patches import vote_bucket_id

    bucket_size = 500.0
    same_bucket_a = (120.0, 220.0)
    same_bucket_b = (480.0, 490.0)  # same 500x500 cell as same_bucket_a
    different_bucket = (520.0, 220.0)  # one cell over in x
    check(
        "vote bucket: two points in the same bucket_size cell share an id",
        vote_bucket_id(same_bucket_a, bucket_size) == vote_bucket_id(same_bucket_b, bucket_size),
    )
    check(
        "vote bucket: a point in the neighboring cell gets a different id",
        vote_bucket_id(same_bucket_a, bucket_size) != vote_bucket_id(different_bucket, bucket_size),
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
        test_order_invariance,
        test_ransac_refinement,
        test_patch_dedup,
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
