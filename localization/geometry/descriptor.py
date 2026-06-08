"""CFBVM-PF 24-D building shape-vector descriptor.

For every Douglas-Peucker vertex ``X`` on the building contours, CFBVM-PF
partitions the local neighbourhood into three concentric annuli
``[0, 20] m``, ``[20, 40] m``, ``[40, 60] m`` and eight 45-degree sectors.
The descriptor is the building area inside each sector:

    f^X = [s11, s12, ..., s18, s21, ..., s28, s31, ..., s38]

The returned areas are in square meters. Polygon coordinates remain in image
pixels; ``meters_per_pixel`` converts both radii and intersected areas.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


VertexXY = Tuple[float, float]

CFBVM_REFERENCE_RADII_M: Tuple[float, float, float] = (20.0, 40.0, 60.0)
CFBVM_SECTOR_COUNT = 8
CFBVM_DESCRIPTOR_DIM = len(CFBVM_REFERENCE_RADII_M) * CFBVM_SECTOR_COUNT
CFBVM_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    f"s{i}{j}"
    for i in range(1, len(CFBVM_REFERENCE_RADII_M) + 1)
    for j in range(1, CFBVM_SECTOR_COUNT + 1)
)

# UAV-VisLoc satellite tiles are treated as approximately 0.3 m/px in the
# existing tutorial/pipeline (100 px stride ~= 30 m). Callers can override this
# when height/GSD calibration supplies a different scale.
DEFAULT_METERS_PER_PIXEL = 0.3


def _dist(p1: VertexXY, p2: VertexXY) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def interior_angles(v0: VertexXY, v1: VertexXY, v2: VertexXY) -> List[float]:
    """Three interior angles of a triangle (deg), sorted ascending.

    Kept as a lightweight geometry helper for external callers; the active
    descriptor pipeline no longer uses triangle/Ekeland features.
    """
    a = _dist(v1, v2)
    b = _dist(v0, v2)
    c = _dist(v0, v1)
    if a * b == 0 or b * c == 0 or c * a == 0:
        return [0.0, 0.0, 0.0]
    try:
        angle_a = math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c)))))
        angle_b = math.degrees(math.acos(max(-1.0, min(1.0, (a * a + c * c - b * b) / (2 * a * c)))))
        angle_c = 180.0 - angle_a - angle_b
    except ValueError:
        return [0.0, 0.0, 0.0]
    return sorted([angle_a, angle_b, angle_c])


def _as_xy_array(polygon_xy: Sequence[Sequence[float]]) -> np.ndarray:
    coords = np.asarray(polygon_xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float64)
    if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def _repaired_polygon(coords: np.ndarray):
    if coords.shape[0] < 3:
        return None
    try:
        geom = Polygon(coords)
    except Exception:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty or geom.area <= 0.0:
        return None
    return geom


def _radii_to_pixels(
    reference_radii_m: Sequence[float],
    meters_per_pixel: float,
) -> np.ndarray:
    radii_m = np.asarray(reference_radii_m, dtype=np.float64)
    if radii_m.ndim != 1 or radii_m.size != 3:
        raise ValueError(f"reference_radii_m must contain exactly 3 radii; got {reference_radii_m!r}")
    if not np.all(np.isfinite(radii_m)) or np.any(radii_m <= 0.0):
        raise ValueError(f"reference_radii_m must be positive finite values; got {reference_radii_m!r}")
    if np.any(np.diff(radii_m) <= 0.0):
        raise ValueError(f"reference_radii_m must be strictly increasing; got {reference_radii_m!r}")
    meters_per_pixel = float(meters_per_pixel)
    if not math.isfinite(meters_per_pixel) or meters_per_pixel <= 0.0:
        raise ValueError(f"meters_per_pixel must be a positive finite value; got {meters_per_pixel!r}")
    return radii_m / meters_per_pixel


def _polar_point(center_xy: VertexXY, radius_px: float, angle_deg: float) -> VertexXY:
    """Image-coordinate polar point; positive angles advance clockwise."""
    theta = math.radians(float(angle_deg))
    return (
        float(center_xy[0]) + float(radius_px) * math.cos(theta),
        float(center_xy[1]) + float(radius_px) * math.sin(theta),
    )


def _sector_polygon(
    center_xy: VertexXY,
    inner_radius_px: float,
    outer_radius_px: float,
    start_angle_deg: float,
    end_angle_deg: float,
    arc_segments: int,
) -> Optional[Polygon]:
    if outer_radius_px <= 0.0 or end_angle_deg <= start_angle_deg:
        return None

    n_segments = max(2, int(arc_segments))
    angles = np.linspace(float(start_angle_deg), float(end_angle_deg), n_segments + 1)
    outer = [_polar_point(center_xy, outer_radius_px, a) for a in angles]
    if inner_radius_px <= 1e-9:
        coords = [tuple(map(float, center_xy))] + outer
    else:
        inner = [_polar_point(center_xy, inner_radius_px, a) for a in angles[::-1]]
        coords = outer + inner

    try:
        sector = Polygon(coords)
    except Exception:
        return None
    if not sector.is_valid:
        sector = sector.buffer(0)
    if sector.is_empty or sector.area <= 0.0:
        return None
    return sector


def building_shape_vector(
    vertex_xy: VertexXY,
    building_geometry,
    reference_radii_m: Sequence[float] = CFBVM_REFERENCE_RADII_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
    arc_segments: int = 12,
) -> np.ndarray:
    """Compute one 24-D CFBVM-PF vector for a contour vertex.

    ``building_geometry`` should represent the union of all building polygons in
    the current image/patch, matching the paper's "distribution of buildings in
    the area" definition rather than a single-building-only descriptor.
    """
    if building_geometry is None or building_geometry.is_empty:
        return np.zeros((CFBVM_DESCRIPTOR_DIM,), dtype=np.float32)

    radii_px = _radii_to_pixels(reference_radii_m, meters_per_pixel)
    area_scale = float(meters_per_pixel) ** 2
    sector_angle = 360.0 / float(CFBVM_SECTOR_COUNT)

    values: List[float] = []
    inner_radius = 0.0
    for outer_radius in radii_px:
        for sector_idx in range(CFBVM_SECTOR_COUNT):
            start = sector_idx * sector_angle
            end = (sector_idx + 1) * sector_angle
            sector = _sector_polygon(
                center_xy=vertex_xy,
                inner_radius_px=float(inner_radius),
                outer_radius_px=float(outer_radius),
                start_angle_deg=start,
                end_angle_deg=end,
                arc_segments=arc_segments,
            )
            if sector is None:
                values.append(0.0)
                continue
            values.append(float(building_geometry.intersection(sector).area) * area_scale)
        inner_radius = float(outer_radius)

    return np.asarray(values, dtype=np.float32)


def building_shape_vectors_from_polygons(
    polygons_xy: Sequence[Sequence[Sequence[float]]],
    reference_radii_m: Sequence[float] = CFBVM_REFERENCE_RADII_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
    arc_segments: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(descriptors (N,24), anchors (N,2))`` for all contour vertices.

    ``polygons_xy`` is the set of Douglas-Peucker building contours in one
    image/patch. Every vertex becomes one descriptor anchor, and every sector
    area is measured against the union of all valid building polygons.
    """
    valid_geometries = []
    anchors: List[VertexXY] = []

    for polygon_xy in polygons_xy:
        coords = _as_xy_array(polygon_xy)
        geom = _repaired_polygon(coords)
        if geom is None:
            continue
        valid_geometries.append(geom)
        anchors.extend((float(x), float(y)) for x, y in coords)

    if not valid_geometries or not anchors:
        return (
            np.zeros((0, CFBVM_DESCRIPTOR_DIM), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    building_geometry = unary_union(valid_geometries)
    if building_geometry.is_empty:
        return (
            np.zeros((0, CFBVM_DESCRIPTOR_DIM), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    descriptors = [
        building_shape_vector(
            vertex_xy=anchor,
            building_geometry=building_geometry,
            reference_radii_m=reference_radii_m,
            meters_per_pixel=meters_per_pixel,
            arc_segments=arc_segments,
        )
        for anchor in anchors
    ]
    return np.vstack(descriptors).astype(np.float32, copy=False), np.asarray(anchors, dtype=np.float32)


def triangle_descriptors_from_polygon(
    polygon_xy: Sequence[Sequence[float]],
    max_depth: int = 4,
    reference_radii_m: Sequence[float] = CFBVM_REFERENCE_RADII_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
    arc_segments: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning CFBVM-PF vectors for one polygon.

    ``max_depth`` is ignored; it remains in the signature so older notebook
    calls that configured Ekeland kernel expansion continue to run.
    """
    _ = max_depth
    return building_shape_vectors_from_polygons(
        [polygon_xy],
        reference_radii_m=reference_radii_m,
        meters_per_pixel=meters_per_pixel,
        arc_segments=arc_segments,
    )
