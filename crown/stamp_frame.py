"""Detection of standard document stamp frames (rectangular border frames).

This module is the production implementation of the algorithm prototyped in
``tests/api/hough.py``.  It locates the rectangular frame lines that surround
the "payload" area of a scanned document page (the stamp frame) and returns the
two largest such frames.

The algorithm:

1. Convert the source image to grayscale and run Canny edge detection.
2. Run the probabilistic Hough line transform (``cv.HoughLinesP``) on the
   (optionally inverted) edge image.
3. Merge overlapping collinear line segments produced by Hough
   (``_merge_collinear_lines``), then deduplicate near-identical survivors
   (``_deduplicate_segments``).
4. Estimate the page skew from the almost-vertical lines
   (``_calc_most_probable_skew_angle``); when the skew is significant, deskew
   the image and re-run steps 1-3 on the rotated image.
5. Keep only the segments whose endpoints lie in the outer border stripe
   (i.e. outside the central "payload" bbox) -- those are the frame lines.
6. Find all pairwise intersections of the surviving segments.
7. Walk the segment-intersection graph with DFS to discover closed rectangular
   loops (cycles of length 4 whose edges are long enough and whose area exceeds
   ``square_threshold``).
8. Derive the frame's bottom-right origin from the loops themselves
   (``_find_bottom_right_corner_of_loops``) and keep the loops that touch it.
9. Select the page-content and metadata/stamp frames with a content-aware
   decision tree: when the second-largest frame is too small to be the real
   stamp, locate the payload bottom (``_find_payload_bottom_y``) or the stamp
   top (``_find_stamp_top_y``) from vertical lines inside the main frame, then
   pick the largest frame entirely below that y-threshold
   (``_find_largest_frame_below_y``).
10. Return the two largest frames by area together with the (possibly deskewed)
    image they are defined in.

All pixel-based tolerances (Hough parameters, dedup/intersection tolerances,
minimum edge length, corner-match tolerance, merge tolerances, square
threshold, payload/stamp search tolerances) are scaled relative to the
reference page size of 2500x3500 pixels, for which the defaults were tuned.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import os
import time
from tracemalloc import start
import cv2 as cv
import numpy as np
from PIL import Image, ImageOps

from crown.settings import crown_settings
from crown.text_detection.east_text_detection import detect_texts


# Reference page dimensions for which the default parameters were tuned.
_REF_W = 2500
_REF_H = 3500
# Scale pixel-based parameters by the geometric mean of the per-dimension
# ratios so that a 2500x3500 page yields a scale of exactly 1.0.
_REF_SCALE = (_REF_W * _REF_H) ** 0.5


@dataclass(frozen=True)
class StampFrame:
    """A detected rectangular stamp frame.

    ``bbox`` is ``(x_min, y_min, x_max, y_max)`` in source-image pixel
    coordinates.  ``area`` is the (integer) product of the two edge lengths
    used for ranking.
    """

    bbox: tuple[int, int, int, int]
    area: int


# ---------------------------------------------------------------------------
# Segment orientation helpers
# ---------------------------------------------------------------------------


def _seg_vertical_size(x1: int, y1: int, x2: int, y2: int) -> int:
    """Vertical extent (abs delta y) of a segment."""
    return abs(y2 - y1)


def _seg_horizontal_size(x1: int, y1: int, x2: int, y2: int) -> int:
    """Horizontal extent (abs delta x) of a segment."""
    return abs(x2 - x1)


def _seg_bottom_y(x1: int, y1: int, x2: int, y2: int) -> int:
    """Larger (lower) y-coordinate of a segment."""
    return max(y1, y2)


def _seg_top_y(x1: int, y1: int, x2: int, y2: int) -> int:
    """Smaller (upper) y-coordinate of a segment."""
    return min(y1, y2)


def _normalize_segment(x1: int, y1: int, x2: int, y2: int) -> list[int]:
    """Normalize a segment so its first endpoint is the canonical anchor.

    For a vertical segment (``|dy| >= |dx|``) the first endpoint carries the
    uppermost (smallest y) point; for a horizontal segment (``|dx| > |dy|``)
    the first endpoint carries the leftmost (smallest x) point.  The returned
    list is ``[x1, y1, x2, y2]`` with the endpoints swapped when needed.
    """
    is_vertical = abs(y2 - y1) >= abs(x2 - x1)
    if is_vertical:
        if y1 > y2:
            return [x2, y2, x1, y1]
    else:
        if x1 > x2:
            return [x2, y2, x1, y1]
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Point/segment geometry (ported from the POC)
# ---------------------------------------------------------------------------


def _calc_point2seg_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[int, int, str]:
    """Distance from point ``(px, py)`` to segment ``(x1,y1)-(x2,y2)``.

    Returns ``(d_tang, d_ort, proj)`` where ``d_tang`` is the tangential
    distance (along the segment) from the nearest endpoint when the projection
    falls outside the segment, ``d_ort`` is the orthogonal (perpendicular)
    distance, and ``proj`` is ``'A'``, ``'B'`` or ``'P'`` indicating whether the
    closest point is the start, the end, or an interior projection.
    """
    ABx = x2 - x1
    ABy = y2 - y1
    APx = px - x1
    APy = py - y1
    is_hor = abs(ABy) < abs(ABx)

    AB_AB = ABx * ABx + ABy * ABy
    if AB_AB == 0:
        # Degenerate segment: distance to the single point.
        return int(np.sqrt(APx * APx + APy * APy)), 0, "A"

    t = (APx * ABx + APy * ABy) / AB_AB

    if t < 0.0:
        if is_hor:
            d_tang = abs(px - x1)
            d_ort = abs(py - y1)
        else:
            d_tang = abs(py - y1)
            d_ort = abs(px - x1)
        return int(d_tang), int(d_ort), "A"
    if t > 1.0:
        if is_hor:
            d_tang = abs(px - x2)
            d_ort = abs(py - y2)
        else:
            d_tang = abs(py - y2)
            d_ort = abs(px - x2)
        return int(d_tang), int(d_ort), "B"

    closest_x = x1 + t * ABx
    closest_y = y1 + t * ABy
    d_ort = np.sqrt((closest_x - px) ** 2 + (closest_y - py) ** 2)
    return 0, int(d_ort), "P"


def _merge_collinear_lines_hv(
    lines: np.ndarray | list[list[int]],
    is_horizontal: bool,
    ort_tolerance: float = 10.0,
    tang_tolerance: float = 200.0,
    angle_tolerance_deg: float = 2.0,
) -> list[list[int]]:
    """Merge overlapping, collinear line segments.

    Two segments are merged when they share the same dominant orientation
    (both horizontal or both vertical), their angular difference is within
    ``angle_tolerance_deg``, and at least one endpoint of one segment lies
    within ``ort_tolerance`` pixels perpendicular and ``tang_tolerance`` pixels
    tangential of the other segment.  The merged segment is the longest
    candidate spanning the two segments' endpoints.
    """
    if lines is None or len(lines) == 0:
        return []

    if is_horizontal:
        cleaned_lines = sorted(
            [_normalize_segment(*seg) for seg in np.array(lines).reshape(-1, 4).tolist()],
            key=lambda x: (int(x[0]), int(x[1])),
        )
    else:
        cleaned_lines = sorted(
            [_normalize_segment(*seg) for seg in np.array(lines).reshape(-1, 4).tolist()],
            key=lambda x: (int(x[1]), int(x[0])),
        )
    absorbed = np.zeros(len(cleaned_lines), dtype=bool)

    for i, line1 in enumerate(cleaned_lines):
        if absorbed[i]:
            continue

        x1, y1, x2, y2 = line1
        angle1 = 0 if is_horizontal else np.pi / 2

        for j, line2 in enumerate(cleaned_lines):
            if j <= i or absorbed[j]:
                continue

            x3, y3, x4, y4 = line2
            angle2 = np.arctan2(abs(y4 - y3), abs(x4 - x3))

            angle_diff = min(
                abs(angle1 - angle2),
                abs(angle1 - angle2 + np.pi),
                abs(angle1 - angle2 - np.pi),
            )
            if angle_diff > np.radians(angle_tolerance_deg):
                continue

            d_tang1, d_ort1, _ = _calc_point2seg_distance(x3, y3, x1, y1, x2, y2)
            a_within = d_ort1 <= ort_tolerance and d_tang1 <= tang_tolerance
            d_tang2, d_ort2, _ = _calc_point2seg_distance(x4, y4, x1, y1, x2, y2)
            b_within = d_ort2 <= ort_tolerance and d_tang2 <= tang_tolerance

            if not (a_within or b_within):
                continue

            candidates = [
                (x1, y1, x2, y2),
                (x3, y3, x4, y4),
                [x1, y1, x4, y4],
                [x3, y3, x2, y2],
                [x1, y1, x3, y3],
                [x2, y2, x4, y4],
            ]
            best_line = []
            max_len = 0
            for c in candidates:
                l = np.hypot(c[2] - c[0], c[3] - c[1])
                if l > max_len:
                    max_len = l
                    best_line = c
            angle2 = np.arctan2(
                abs(best_line[3] - best_line[1]), abs(best_line[2] - best_line[0])
            )
            angle_diff = min(
                abs(angle1 - angle2),
                abs(angle1 - angle2 + np.pi),
                abs(angle1 - angle2 - np.pi),
            )
            if angle_diff > np.radians(angle_tolerance_deg):
                continue
            cleaned_lines[i] = best_line
            x1, y1, x2, y2 = best_line
            absorbed[j] = True

    return [line for i, line in enumerate(cleaned_lines) if not absorbed[i]]


def _merge_collinear_lines(
    lines: np.ndarray | list[list[int]],
    ort_tolerance: float = 10.0,
    tang_tolerance: float = 200.0,
    angle_tolerance_deg: float = 2.0,
) -> list[list[int]]:
    """Merge overlapping, collinear line segments.

    Two segments are merged when they share the same dominant orientation
    (both horizontal or both vertical), their angular difference is within
    ``angle_tolerance_deg``, and at least one endpoint of one segment lies
    within ``ort_tolerance`` pixels perpendicular and ``tang_tolerance`` pixels
    tangential of the other segment.  The merged segment is the longest
    candidate spanning the two segments' endpoints.
    """
    if lines is None or len(lines) == 0:
        return []

    start = time.perf_counter()
    min_length = tang_tolerance / 2
    cleaned_segments = [seg for seg in np.array(lines).reshape(-1, 4).tolist()
                     if np.hypot(seg[2] - seg[0], seg[3] - seg[1]) >= min_length]
    horizontal_lines = [seg for seg in cleaned_segments if abs(seg[3] - seg[1]) < abs(seg[2] - seg[0])]
    vertical_lines = [seg for seg in cleaned_segments if abs(seg[3] - seg[1]) >= abs(seg[2] - seg[0])]
    merged_horizontal = _merge_collinear_lines_hv(
        horizontal_lines, True, ort_tolerance, tang_tolerance, angle_tolerance_deg
    )
    merged_vertical = _merge_collinear_lines_hv(
        vertical_lines, False, ort_tolerance, tang_tolerance, angle_tolerance_deg
    )
    ret: list[list[int]] = merged_horizontal + merged_vertical
    end = time.perf_counter()
    print(f"_merge_collinear_lines took {end - start:.6f} seconds")
    return ret


# ---------------------------------------------------------------------------
# Skew estimation (ported from the POC)
# ---------------------------------------------------------------------------


def _calc_most_probable_skew_angle(
    lines: list[list],
    max_count: int = 20,
    vertical_tolerance_deg: float = 20.0,
    bin_size_deg: float = 0.5,
) -> tuple[float, list[list[int]]]:
    """Most probable signed skew angle (degrees) from almost-vertical lines.

    Selects the ``max_count`` longest almost-vertical segments, weights them by
    length, accumulates them into angular bins of width ``bin_size_deg`` and
    returns the center of the heaviest bin.  ``0.0`` when no suitable lines are
    found.  The returned angle can be used directly with
    ``cv.getRotationMatrix2D`` to deskew the image.
    """
    if lines is None:
        return 0.0, []

    segments = np.asarray(lines).reshape(-1, 4)
    if segments.size == 0:
        return 0.0, []

    tol_rad = np.radians(vertical_tolerance_deg)
    candidates: list[tuple[float, float, list[int]]] = []
    for x1, y1, x2, y2 in segments.tolist():
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue
        angle_from_vertical = float(np.arctan2(dx, dy))
        if angle_from_vertical > np.pi / 2:
            angle_from_vertical -= np.pi
        elif angle_from_vertical < -np.pi / 2:
            angle_from_vertical += np.pi
        if abs(angle_from_vertical) > tol_rad:
            continue
        candidates.append((length, angle_from_vertical, [x1, y1, x2, y2]))

    if not candidates:
        return 0.0, []

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_count]

    angles_deg = np.degrees([a for _, a, _ in selected])
    weights = np.array([l for l, _, _ in selected], dtype=float)

    if bin_size_deg <= 0:
        bin_size_deg = 0.5
    min_angle = float(np.min(angles_deg))
    max_angle = float(np.max(angles_deg))
    left = min_angle - bin_size_deg / 2
    right = max_angle + bin_size_deg / 2
    n_bins = max(int(np.ceil((right - left) / bin_size_deg)), 1)
    edges = np.linspace(left, left + n_bins * bin_size_deg, n_bins + 1)

    hist, edges = np.histogram(angles_deg, bins=edges, weights=weights)
    best_bin = int(np.argmax(hist))
    skew_angle_deg = float((edges[best_bin] + edges[best_bin + 1]) / 2.0)

    selected_lines = [seg for _, _, seg in selected]
    return skew_angle_deg, selected_lines


# ---------------------------------------------------------------------------
# Bbox / endpoint tests (ported from the POC)
# ---------------------------------------------------------------------------


def _is_endpoint_outside_bbox(
    p1: tuple[int, int],
    p2: tuple[int, int],
    bbox: tuple[int, int, int, int],
    tolerance: int = 0,
) -> bool:
    """True if at least one endpoint lies outside ``bbox``.

    ``tolerance`` expands the bbox symmetrically before the test, so a point
    within ``tolerance`` pixels of the boundary still counts as inside.
    """
    x1, y1 = p1
    x2, y2 = p2
    min_x, min_y, max_x, max_y = bbox
    p1_outside = (
        x1 < min_x - tolerance
        or x1 > max_x + tolerance
        or y1 < min_y - tolerance
        or y1 > max_y + tolerance
    )
    p2_outside = (
        x2 < min_x - tolerance
        or x2 > max_x + tolerance
        or y2 < min_y - tolerance
        or y2 > max_y + tolerance
    )
    return p1_outside or p2_outside


def _is_endpoint_inside_bbox(
    p1: tuple[int, int],
    p2: tuple[int, int],
    bbox: tuple[int, int, int, int],
    tolerance: int = 0,
) -> bool:
    """True if at least one endpoint lies inside ``bbox`` (with ``tolerance``)."""
    x1, y1 = p1
    x2, y2 = p2
    min_x, min_y, max_x, max_y = bbox
    p1_inside = (
        x1 >= min_x - tolerance
        and x1 <= max_x + tolerance
        and y1 >= min_y - tolerance
        and y1 <= max_y + tolerance
    )
    p2_inside = (
        x2 >= min_x - tolerance
        and x2 <= max_x + tolerance
        and y2 >= min_y - tolerance
        and y2 <= max_y + tolerance
    )
    return p1_inside or p2_inside


# ---------------------------------------------------------------------------
# Segment intersection (ported from the POC)
# ---------------------------------------------------------------------------


def _get_segment_intersection(
    indexed_linesseg: dict[int, list[int]],
    ix1: int,
    ix2: int,
    tolerance: float = 5.0,
) -> tuple[frozenset[int], int, int] | None:
    """
    Finds the intersection point of two line segments with a given endpoint tolerance.
    
    line1, line2: Format ((x1, y1), (x2, y2))
    tolerance: Number of pixels allowed beyond the actual segment boundaries.
    """
    seg1 = indexed_linesseg[ix1]
    seg2 = indexed_linesseg[ix2]
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2
    seg1_is_horizontal = abs(y2 - y1) < abs(x2 - x1)
    seg2_is_horizontal = abs(y4 - y3) < abs(x4 - x3)
    if seg1_is_horizontal == seg2_is_horizontal:
        return None  # Both segments are horizontal or both are vertical; no intersection

    # Line equation coefficients: Ax + By = C
    # This avoids divide-by-zero errors for perfectly vertical lines
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    # Calculate determinant
    determinant = a1 * b2 - a2 * b1

    if abs(determinant) < 1e-9:
        return None  # The lines are parallel or collinear

    # Cramer's Rule to find intersection point (px, py)
    px = (b2 * c1 - b1 * c2) / determinant
    py = (a1 * c2 - a2 * c1) / determinant

    # Tolerance helper to check if point falls within bounding box
    def is_on_segment(val, endpoint_a, endpoint_b, tol):
        return (min(endpoint_a, endpoint_b) - tol) <= val <= (max(endpoint_a, endpoint_b) + tol)

    # Validate that the intersection is inside the bounding bounds of BOTH segments
    if (is_on_segment(px, x1, x2, tolerance) and is_on_segment(py, y1, y2, tolerance) and
        is_on_segment(px, x3, x4, tolerance) and is_on_segment(py, y4, y3, tolerance)):
        return (frozenset({ix1, ix2}), int(round(px)), int(round(py)))
        
    return None


def _find_all_segment_intersections(
    indexed_linesseg: dict[int, list[int]],
    tolerance: float = 5.0,
) -> dict[frozenset[int], tuple[int, int]]:
    start = time.perf_counter()
    intersections: dict[frozenset[int], tuple[int, int]] = {}
    for ix1, ix2 in itertools.combinations(indexed_linesseg.keys(), 2):
        key = frozenset({ix1, ix2})
        if key in intersections:
            continue
        intersection = _get_segment_intersection(
            indexed_linesseg, ix1, ix2, tolerance
        )
        if intersection is not None:
            _, x, y = intersection
            intersections[key] = (x, y)
    end = time.perf_counter()
    print(f"_find_all_segment_intersections took {end - start:.6f} seconds")
    return intersections


# ---------------------------------------------------------------------------
# Segment deduplication (ported from the POC, unchanged in behaviour)
# ---------------------------------------------------------------------------


def _deduplicate_segments_hv(
    segments: list[list[int]] | np.ndarray,
    is_horizontal: bool,
    tolerance: float = 5.0,
) -> list[list[int]]:
    """Collapse near-identical Hough segments of a single orientation.

    ``segments`` are expected to share the same dominant orientation (all
    horizontal when ``is_horizontal`` is True, all vertical otherwise).  Each
    segment is first normalized with [`_normalize_segment()`](crown/stamp_frame.py:103)
    so that the first endpoint is the canonical anchor (leftmost for
    horizontal, uppermost for vertical).  Because matching segments then share
    the same endpoint ordering, [`_endpoints_close()`](crown/stamp_frame.py)
    only needs to compare the first endpoints to the first endpoints and the
    second to the second -- no second cross-ordered distance computation is
    required.
    """
    if segments is None or len(segments) == 0:
        return []

    tol_sq = tolerance * tolerance
    if is_horizontal:
        cleaned = sorted(
            [_normalize_segment(*seg) for seg in np.array(segments).reshape(-1, 4).tolist()],
            key=lambda x: (int(x[1]), int(x[0])),
        )
    else:
        cleaned = sorted(
            [_normalize_segment(*seg) for seg in np.array(segments).reshape(-1, 4).tolist()],
            key=lambda x: (int(x[0]), int(x[1])),
        )

    def endpoints_close(a: list[int], b: list[int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        if abs((ax1 - bx1)) > tolerance or abs((ay1 - by1)) > tolerance:
            return False
        d1 = (ax1 - bx1) ** 2 + (ay1 - by1) ** 2
        if d1 > tol_sq:
            return False
        if abs((ax2 - bx2)) > tolerance or abs((ay2 - by2)) > tolerance:
            return False
        d2 = (ax2 - bx2) ** 2 + (ay2 - by2) ** 2
        return d2 <= tol_sq

    groups: list[list[list[int]]] = []
    for seg in cleaned:
        for group in groups:
            if endpoints_close(group[0], seg):
                group.append(seg)
                break
        else:
            groups.append([seg])

    averaged: list[list[int]] = []
    for group in groups:
        n = len(group)
        sx1 = sum(s[0] for s in group) / n
        sy1 = sum(s[1] for s in group) / n
        sx2 = sum(s[2] for s in group) / n
        sy2 = sum(s[3] for s in group) / n
        averaged.append(
            [int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2))]
        )
    return averaged


def _deduplicate_segments(
    segments: list[list[int]] | np.ndarray,
    tolerance: float = 5.0,
) -> list[list[int]]:
    """Collapse near-identical Hough segments into single averaged segments.

    Splits the input into horizontal and vertical segments and deduplicates
    each orientation separately via [`_deduplicate_segments_hv()`](crown/stamp_frame.py),
    mirroring the structure of [`_merge_collinear_lines()`](crown/stamp_frame.py:266).
    """
    if (isinstance(segments, np.ndarray) and segments.size == 0) or (not isinstance(segments, np.ndarray) and not segments):
        return []
    start = time.perf_counter()
    cleaned_segments = np.array(segments).reshape(-1, 4).tolist()
    horizontal_segments = [seg for seg in cleaned_segments if abs(seg[3] - seg[1]) < abs(seg[2] - seg[0])]
    vertical_segments = [seg for seg in cleaned_segments if abs(seg[3] - seg[1]) >= abs(seg[2] - seg[0])]
    dedup_horizontal = _deduplicate_segments_hv(
        horizontal_segments, True, tolerance
    )
    dedup_vertical = _deduplicate_segments_hv(
        vertical_segments, False, tolerance
    )
    ret: list[list[int]] = dedup_horizontal + dedup_vertical
    end = time.perf_counter()
    print(f"_deduplicate_segments took {end - start:.6f} seconds")
    return ret


# ---------------------------------------------------------------------------
# Loop discovery (ported from the POC, with square_threshold)
# ---------------------------------------------------------------------------


def get_edge_length(start_xy: tuple[int, int], end_xy: tuple[int, int]) -> int:
    x1, y1 = start_xy
    x2, y2 = end_xy
    return max(abs(x2 - x1), abs(y2 - y1))


def _loop_bottom_right_corner(
    loop: list[tuple[int, tuple[int, int]]],
) -> tuple[int, int]:
    """Bottom-right corner of a single loop.

    The bottom-right corner is the corner point with the greatest ``x + y``
    sum; ties are broken by the greatest ``x`` and then the greatest ``y``.
    """
    best_point = loop[0][1]
    best_key = (best_point[0] + best_point[1], best_point[0], best_point[1])
    for _, (x, y) in loop:
        key = (x + y, x, y)
        if key > best_key:
            best_key = key
            best_point = (x, y)
    return best_point


def _frame_sort_key(
    frame: tuple[list[tuple[int, tuple[int, int]]], int],
    origin_point: tuple[int, int] | None,
    corner_tolerance: float,
) -> float:
    """Sort key for ranking detected frames by area with an origin bonus.

    The base key is the frame's area (``frame[1]``).  Frames whose bottom-right
    corner is close to ``origin_point`` get a multiplicative boost of up to
    15%: a corner exactly at ``origin_point`` (distance 0) receives the full
    1.15 factor, the factor decreases linearly to 1.0 as the distance grows to
    ``corner_tolerance``, and stays at 1.0 for any larger distance.  When
    ``origin_point`` is ``None`` the plain area is returned.
    """
    area = frame[1]
    if origin_point is None or corner_tolerance <= 0:
        return float(area)
    corner = _loop_bottom_right_corner(frame[0])
    dist = float(np.hypot(corner[0] - origin_point[0], corner[1] - origin_point[1]))
    if dist >= corner_tolerance:
        return float(area)
    boost = 1.0 + 0.15 * (1.0 - dist / corner_tolerance)
    return area * boost


def _find_bottom_right_corner_of_loops(
    closed_loops: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]],
    edge_len_threshold: int = 100,
    corner_tolerance: float = 20.0,
) -> tuple[int, int] | None:
    """Most bottom-right corner among all loops' corner points.

    The "most bottom-right" corner is the one with the greatest ``x + y`` sum;
    ties are broken by the greatest ``x`` and then the greatest ``y``.

    After the candidate point is found, every loop whose own bottom-right
    corner (see [`_loop_bottom_right_corner()`](crown/stamp_frame.py)) lies
    within ``edge_len_threshold`` pixels of the candidate is collected.  The
    collected corners are then clustered with ``corner_tolerance`` as the
    cluster radius: the largest cluster is selected and its centroid (the
    average of its members) is returned as the most probable origin point.
    When only one corner is collected, the candidate point itself is returned.
    """
    best_point: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for loop, _ in closed_loops.values():
        for _, (x, y) in loop:
            key = (x + y, x, y)
            if best_key is None or key > best_key:
                best_key = key
                best_point = (x, y)
    if best_point is None:
        return None

    # Collect the bottom-right corner of every loop that lies within
    # ``edge_len_threshold`` of the candidate point.
    nearby_corners: list[tuple[int, int]] = []
    for loop, _ in closed_loops.values():
        corner = _loop_bottom_right_corner(loop)
        if np.hypot(corner[0] - best_point[0], corner[1] - best_point[1]) <= edge_len_threshold:
            nearby_corners.append(corner)

    if len(nearby_corners) > 1:
        # Cluster the collected corners using ``corner_tolerance`` as the
        # cluster radius and pick the most populated cluster.  Greedy
        # single-linkage clustering: a corner joins a cluster when it is within
        # ``corner_tolerance`` of any member already in it.
        max_sq = corner_tolerance * corner_tolerance
        clusters: list[list[tuple[int, int]]] = []
        for corner in nearby_corners:
            placed = False
            for cluster in clusters:
                for member in cluster:
                    dx = corner[0] - member[0]
                    dy = corner[1] - member[1]
                    if dx * dx + dy * dy <= max_sq:
                        cluster.append(corner)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([corner])
        # Select the largest cluster; ties broken by the cluster whose centroid
        # is closest to the candidate point.
        best_cluster: list[tuple[int, int]] | None = None
        best_cluster_key: tuple[int, float] | None = None
        for cluster in clusters:
            cx = sum(c[0] for c in cluster) / len(cluster)
            cy = sum(c[1] for c in cluster) / len(cluster)
            dist = (cx - best_point[0]) ** 2 + (cy - best_point[1]) ** 2
            key = (len(cluster), -dist)
            if best_cluster_key is None or key > best_cluster_key:
                best_cluster_key = key
                best_cluster = cluster
        if best_cluster is not None and len(best_cluster) > 1:
            ax = sum(c[0] for c in best_cluster) / len(best_cluster)
            ay = sum(c[1] for c in best_cluster) / len(best_cluster)
            return (int(round(ax)), int(round(ay)))

    return best_point


def _path_has_points(
    path: list[tuple[int, tuple[int, int]]],
    points: set[tuple[int, int]],
    tolerance: int,
) -> bool:
    for _, point in path:
        for px, py in points:
            if abs(point[0] - px) <= tolerance and abs(point[1] - py) <= tolerance:
                return True
    return False


def _find_all_closed_loops(
    indexed_linesseg: dict[int, list[int]],
    intersections: dict[frozenset[int], tuple[int, int]],
    edge_len_threshold: int = 100,
    square_threshold: int | None = None,
) -> dict[frozenset[int], tuple[list[tuple[int, tuple[int, int]]], int]]:
    """Finds all closed loops formed by the line segments."""
    start = time.perf_counter()
    # Build adjacency list for the graph of line segments
    adjacency: dict[int, dict[int, tuple[int, int]]] = {
        ix: dict() for ix in indexed_linesseg.keys()
    }
    for ix1, ix2 in intersections.keys():
        adjacency[ix1][ix2] = intersections[frozenset({ix1, ix2})]
        adjacency[ix2][ix1] = intersections[frozenset({ix1, ix2})]
    # loops: list[list[tuple[int, tuple[int, int]]]] = []
    unique_loops: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]] = {}
    if square_threshold is None:
        square_threshold = edge_len_threshold * edge_len_threshold

    # Use DFS to find all cycles in the graph
    def dfs(
        current: int,
        start: int,
        visited: set[int],
        parent_path: list[tuple[int, tuple[int, int]]],
    ) -> None:
        if len(parent_path) >= 4:  # Only consider cycles of length 4 (rectangles)
            return
        visited.add(current)
        for neighbor, point in adjacency[current].items():
            path = parent_path.copy()
            path.append((current, point))
            if len(path) > 2:
                l1 = get_edge_length(path[0][1], path[1][1])
                l2 = get_edge_length(path[1][1], path[2][1])
                if (
                    l1 < edge_len_threshold or l2 < edge_len_threshold
                ):  # Skip if any edge is too short
                    continue
            if neighbor == start and len(path) == 4:
                # Found a cycle
                square: int = l1 * l2 if l1 and l2 else 0
                if square < square_threshold:
                    continue
                seg_set = frozenset([segment for segment, _ in path])
                if seg_set not in unique_loops:
                    unique_loops[seg_set] = (path, square)
                # loops.append(path)
            elif neighbor not in visited:
                dfs(neighbor, start, visited, path)

        visited.remove(current)

    for node in indexed_linesseg.keys():
        dfs(node, node, set(), [])
    end = time.perf_counter()
    print(f"_find_all_closed_loops took {end - start:.6f} seconds")

    return unique_loops


# ---------------------------------------------------------------------------
# Content-aware frame selection helpers (ported from the POC)
# ---------------------------------------------------------------------------


def _loop_bbox(
    loop: list[tuple[int, tuple[int, int]]],
) -> tuple[int, int, int, int]:
    """Bounding box ``(x1, y1, x2, y2)`` of a loop's corner points."""
    x_coords = [point[1][0] for point in loop]
    y_coords = [point[1][1] for point in loop]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def _loop_vertical_border_indexes(
    loop: list[tuple[int, tuple[int, int]]],
) -> list[int]:
    """Indexes of the loop edges that are (dominantly) vertical."""
    frame_v_borders: list[int] = []
    prev_point = loop[-1][1]
    for item in loop:
        index, (x, y) = item
        x1, y1 = prev_point
        x2, y2 = x, y
        if _seg_vertical_size(x1, y1, x2, y2) > _seg_horizontal_size(x1, y1, x2, y2):
            frame_v_borders.append(index)
        prev_point = (x, y)
    return frame_v_borders


def _find_payload_bottom_y(
    indexed_lines: dict[int, list[int]],
    standard_frames: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]],
    inside_tolerance: int = 10,
    min_bottom_gap: int = 200,
) -> tuple[int | None, list[int]]:
    """Bottom y of vertical lines inside the first standard frame whose upper
    end is above the frame's mid-height.

    Used to locate the bottom edge of the page payload (the text area) so the
    stamp frame below it can be found.  Returns
    ``(bottom_y + inside_tolerance, segment_indexes)`` or ``(None, [])``.
    """
    if not standard_frames:
        return None, []

    loop, _ = next(iter(standard_frames.values()))
    frame_x1, frame_y1, frame_x2, frame_y2 = _loop_bbox(loop)
    frame_height = frame_y2 - frame_y1
    if frame_height <= 0:
        return None, []
    min_length = frame_height / 10

    frame_v_borders = _loop_vertical_border_indexes(loop)

    segment_indexes: list[int] = []
    bottom_ys: list[int] = []
    for index, seg in indexed_lines.items():
        if index in frame_v_borders:
            continue
        x1, y1, x2, y2 = seg
        if abs(max(y1, y2) - frame_y2) <= min_bottom_gap or max(y1, y2) >= frame_y2:
            continue
        if _seg_vertical_size(x1, y1, x2, y2) <= _seg_horizontal_size(x1, y1, x2, y2):
            continue
        if _is_endpoint_outside_bbox(
            (x1, y1), (x2, y2),
            (frame_x1, frame_y1, frame_x2, frame_y2),
            tolerance=inside_tolerance,
        ):
            continue
        if not _is_endpoint_inside_bbox(
            (x1, y1), (x2, y2),
            (frame_x1, frame_y1, frame_x2, frame_y2 // 2),
            tolerance=inside_tolerance,
        ):
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length <= min_length:
            continue
        segment_indexes.append(index)
        bottom_ys.append(_seg_bottom_y(x1, y1, x2, y2))

    if not bottom_ys:
        return None, []

    bottom_ys = sorted(bottom_ys, reverse=True)
    return bottom_ys[0] + inside_tolerance, segment_indexes


def _find_stamp_top_y(
    indexed_lines: dict[int, list[int]],
    standard_frames: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]],
    inside_tolerance: int = 10,
    max_length_ratio: float = 0.25,
) -> tuple[int | None, list[int]]:
    """Top y of short vertical lines touching the bottom of the first standard
    frame.

    Used to locate the top edge of the stamp frame sitting just below the
    payload.  Returns ``(top_y - inside_tolerance, segment_indexes)`` or
    ``(None, [])``.
    """
    if not standard_frames:
        return None, []

    loop, _ = next(iter(standard_frames.values()))
    frame_x1, frame_y1, frame_x2, frame_y2 = _loop_bbox(loop)
    frame_height = frame_y2 - frame_y1
    if frame_height <= 0:
        return None, []
    max_length = frame_height * max_length_ratio

    frame_v_borders = _loop_vertical_border_indexes(loop)

    segment_indexes: list[int] = []
    top_ys: list[int] = []
    for index, seg in indexed_lines.items():
        if index in frame_v_borders:
            continue
        x1, y1, x2, y2 = seg
        if _seg_vertical_size(x1, y1, x2, y2) <= _seg_horizontal_size(x1, y1, x2, y2):
            continue
        if _is_endpoint_outside_bbox(
            (x1, y1), (x2, y2),
            (frame_x1, frame_y1, frame_x2, frame_y2),
            tolerance=inside_tolerance,
        ):
            continue
        if abs(_seg_bottom_y(x1, y1, x2, y2) - frame_y2) > inside_tolerance:
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= max_length:
            continue
        segment_indexes.append(index)
        top_ys.append(_seg_top_y(x1, y1, x2, y2))

    if not top_ys:
        return None, []

    top_ys = sorted(top_ys)
    return top_ys[0] - inside_tolerance, segment_indexes


def _find_largest_frame_below_y(
    main_frames: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]],
    y_threshold: int,
    inclusive: bool = False,
) -> tuple[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]] | None:
    """Largest (by area) frame whose top edge lies below ``y_threshold``.

    With ``inclusive=True`` a frame whose top edge equals ``y_threshold`` also
    qualifies; otherwise it must be strictly below.
    """
    best_entry: (
        tuple[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]] | None
    ) = None
    for key, (loop, square) in main_frames.items():
        top_y = min(point[1][1] for point in loop)
        if inclusive:
            qualifies = top_y >= y_threshold
        else:
            qualifies = top_y > y_threshold
        if not qualifies:
            continue
        if best_entry is None or square > best_entry[1][1]:
            best_entry = (key, (loop, square))
    return best_entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_stamp_frames(
    image: Image.Image,
    *,
    max_frames: int = 2,
    border_stripe_ratio: float = 0.1,
    canny_low: int = 50,
    canny_high: int = 200,
    canny_aperture: int = 3,
    invert_before_hough: bool | None = None,
    hough_threshold: int = 100,
    hough_min_line_length: int = 100,
    hough_max_line_gap: int = 20,
    dedup_tolerance: float = 30.0,
    intersection_tolerance: float = 20.0,
    edge_len_threshold: int = 100,
    square_threshold: int = 200_000,
    corner_tolerance: float = 20.0,
    merge_ort_tolerance: float = 20.0,
    merge_tang_tolerance: float = 100.0,
    merge_angle_tolerance_deg: float = 5.0,
    skew_vertical_tolerance_deg: float = 5.0,
    skew_min_angle_deg: float = 0.1,
    payload_inside_tolerance: float = 10.0,
    payload_min_bottom_gap: int = 200,
    stamp_max_length_ratio: float = 0.25,
    metadata_min_area_ratio: float = 0.2,
    stamp_max_area_ratio: float = 0.25,
    dpi: tuple[int, int] | None = None,
) -> tuple[list[StampFrame], int, Image.Image,
            float,
            dict[frozenset[int],tuple[list[tuple[int, tuple[int, int]]], int]] | None,
            dict[frozenset[int],tuple[list[tuple[int, tuple[int, int]]], int]] | None,
            dict[int, list[int]] | None,
            tuple[int, int] | None
            ]:
    """Detect the standard rectangular stamp frames on a document page.

    The source is a PIL image (any mode); it is converted to grayscale
    internally.  All pixel-based tolerances are scaled relative to a reference
    page size of 2500x3500 pixels, for which the defaults were tuned, so the
    same defaults work across different scan resolutions.

    The pipeline runs two Hough passes: the first estimates the page skew from
    the almost-vertical lines; when the skew is significant the image is
    deskewed and a second pass is run on the rotated image.  Frame selection is
    content-aware: when the second-largest frame is too small to be the real
    stamp, the payload bottom (or the stamp top) is located from vertical lines
    inside the main frame and the largest frame entirely below that y-threshold
    is chosen as the stamp.

    Parameters
    ----------
    image:
        Source page as a PIL image.
    max_frames:
        Maximum number of frames to return (the largest by area).
    border_stripe_ratio:
        Fraction of each side treated as the outer border stripe.  Segments
        whose endpoints fall inside the central payload bbox are discarded.
    canny_low, canny_high, canny_aperture:
        Canny edge detector parameters.
    invert_before_hough:
        When True, run ``cv.bitwise_not`` on the grayscale image before Hough,
        which suits dark-text-on-light scans.  When ``None`` (the default), the
        document background color is estimated with
        [`_background_color()`](crown/stamp_frame.py) and the image is inverted
        automatically when that background is light (luminance above 127),
        which is the common case for scanned documents.
    hough_threshold, hough_min_line_length, hough_max_line_gap:
        ``cv.HoughLinesP`` parameters.
    dedup_tolerance:
        Endpoint tolerance for collapsing duplicate Hough segments.
    intersection_tolerance:
        Tolerance (pixels) for accepting a segment-segment intersection.
    edge_len_threshold:
        Minimum edge length for a loop edge to be considered.
    square_threshold:
        Minimum loop area (product of the first two edge lengths).
    corner_tolerance:
        Tolerance (pixels) for matching a loop corner to the bottom-right
        origin point of the frame.
    merge_ort_tolerance, merge_tang_tolerance, merge_angle_tolerance_deg:
        Parameters for ``_merge_collinear_lines``.
    skew_vertical_tolerance_deg:
        Maximum deviation from vertical for a line to count toward skew.
    skew_min_angle_deg:
        Skew angles below this are ignored (no deskew).
    payload_inside_tolerance:
        Bbox margin (pixels) for the payload/stamp vertical-line search.
    payload_min_bottom_gap:
        Minimum distance (pixels) from the frame bottom for a payload line.
    stamp_max_length_ratio:
        Max stamp-touching line length as a fraction of the frame height.
    metadata_min_area_ratio:
        When the 2nd frame's area exceeds ``vol1 * metadata_min_area_ratio``
        it is considered not to be the stamp and is replaced.
    stamp_max_area_ratio:
        A candidate stamp is accepted only when its area is below
        ``vol1 * stamp_max_area_ratio``.

    Returns
    -------
    tuple
        ``(frames, corner_tolerance, deskewed_image)`` where ``frames`` is a
        list of up to ``max_frames`` :class:`StampFrame` sorted by area
        descending, ``corner_tolerance`` is the scaled corner tolerance, and
        ``deskewed_image`` is the (possibly rotated) RGB image the frames are
        defined in -- crop from it, not from the original image.
    """
    if dpi is None:
        dpi = image.info.get("dpi", (300, 300))
    rgb = image.convert("RGB")
    w_img, h_img = rgb.size

    # Auto-detect whether to invert before Hough when the caller did not
    # specify.  A light document background (the common scan case) means the
    # frame lines are dark, so inverting yields bright lines on a dark
    # background, which is what the Hough/Canny pipeline expects.
    if invert_before_hough is None:
        bg = _background_color(rgb)
        bg_luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        invert_before_hough = bg_luminance > 127

    # Scale all pixel-based parameters relative to the reference page.
    # scale = ((w_img * h_img) / (_REF_W * _REF_H)) ** 0.5
    scale = dpi[0] / 300.0
    scale = max(scale, 1e-3)

    hough_threshold = max(1, int(round(hough_threshold * scale)))
    hough_min_line_length = max(1, int(round(hough_min_line_length * scale)))
    hough_max_line_gap = max(1, int(round(hough_max_line_gap * scale)))
    dedup_tolerance = dedup_tolerance * scale
    intersection_tolerance = intersection_tolerance * scale
    edge_len_threshold = max(1, int(round(edge_len_threshold * scale)))
    square_threshold = max(1, int(round(square_threshold * scale * scale)))
    corner_tolerance = max(1, int(round(corner_tolerance * scale)))
    merge_ort_tolerance = merge_ort_tolerance * scale
    merge_tang_tolerance = merge_tang_tolerance * scale
    payload_inside_tolerance = payload_inside_tolerance * scale
    payload_min_bottom_gap = max(1, int(round(payload_min_bottom_gap * scale)))

    border_stripe_h = int(h_img * border_stripe_ratio)
    border_stripe_w = int(w_img * border_stripe_ratio)
    payload_bbox = (
        border_stripe_w,
        border_stripe_h,
        w_img - border_stripe_w,
        h_img - border_stripe_h,
    )

    def _hough_pass(
        src_gray: np.ndarray, rho: float, theta_step_deg: float
    ) -> list[list[int]]:
        # hough_src = cv.Canny(src_gray, canny_low, canny_high, None, canny_aperture)
        start = time.perf_counter()
        hough_src = src_gray.copy()
        if invert_before_hough:
            inverted = cv.bitwise_not(hough_src)
            hough_src = inverted
        lines_p = cv.HoughLinesP(
            hough_src,
            rho,
            np.pi / 180 * theta_step_deg,
            hough_threshold,
            None,
            hough_min_line_length,
            hough_max_line_gap,
        )
        print(f"HoughLinesP took {time.perf_counter() - start:.6f} seconds")
        if lines_p is None:
            return []
        lines_p = _deduplicate_segments(lines_p, tolerance=dedup_tolerance)
        merged = _merge_collinear_lines(
            lines_p,
            ort_tolerance=merge_ort_tolerance,
            tang_tolerance=merge_tang_tolerance,
            angle_tolerance_deg=merge_angle_tolerance_deg,
        )
        # merged = _deduplicate_segments(merged, tolerance=dedup_tolerance)
        end = time.perf_counter()
        print(f"_hough_pass took {end - start:.6f} seconds")
        return merged

    gray_pil = rgb.convert("L")
    src_gray = np.array(gray_pil)

    # Pass 1: estimate skew.
    pass1_lines = _hough_pass(src_gray, rho=1, theta_step_deg=1)
    skew_angle, _ = _calc_most_probable_skew_angle(
        pass1_lines, vertical_tolerance_deg=skew_vertical_tolerance_deg
    )

    deskewed = rgb
    if abs(skew_angle) > skew_min_angle_deg:
        # Deskew the RGB image so all downstream crops share its coordinate
        # space.  PIL rotates counter-clockwise for positive angles, so negate
        # the detected skew to correct it.
        # deskewed = rgb.rotate(-skew_angle, expand=False, resample=Image.Resampling.BICUBIC)
        print(f"Deskewing by {skew_angle:.2f} degrees")
        deskewed = rgb.rotate(-skew_angle, expand=False, resample=Image.Resampling.NEAREST)
        gray_pil = deskewed.convert("L")
        if crown_settings.DEBUG_FOLDER:
            os.makedirs(crown_settings.DEBUG_FOLDER, exist_ok=True)
            gray_pil.save(f"{crown_settings.DEBUG_FOLDER}/debug_deskewed.png")
        # bw_pil = gray_pil.convert('1')
        # src_gray = np.array(bw_pil).astype('uint8')
        # src_gray = np.dstack([thresh, thresh, thresh])
        src_gray = np.array(gray_pil)
        # src_gray = np.array(deskewed)
        # Pass 2 on the deskewed image with a coarser accumulator.
        merged_lines = _hough_pass(src_gray, rho=3, theta_step_deg=1)
    else:
        merged_lines = pass1_lines

    lines = _deduplicate_segments(merged_lines, tolerance=dedup_tolerance)

    indexed_lines: dict[int, list[int]] = (
        dict(enumerate(lines)) if lines else {}
    )
    if not indexed_lines:
        return [], corner_tolerance, deskewed, skew_angle, None, None, None, None

    outer_lines = {
        ix: seg
        for ix, seg in indexed_lines.items()
        if _is_endpoint_outside_bbox(
            (seg[0], seg[1]), (seg[2], seg[3]), payload_bbox, tolerance=1
        )
    }
    if not outer_lines:
        return [], corner_tolerance, deskewed, skew_angle, None, None, None, None

    intersections = _find_all_segment_intersections(
        outer_lines, tolerance=intersection_tolerance
    )
    closed_loops = _find_all_closed_loops(
        outer_lines, intersections,
        edge_len_threshold=edge_len_threshold,
        square_threshold=square_threshold,
    )

    origin_point = _find_bottom_right_corner_of_loops(
        closed_loops,
        edge_len_threshold=edge_len_threshold,
        corner_tolerance=corner_tolerance,
    )
    origin_points = {origin_point} if origin_point is not None else set()
    main_frames = {
        ix: val
        for ix, val in closed_loops.items()
        if _path_has_points(val[0], origin_points, tolerance=corner_tolerance)
    }
    main_frames = dict(
        sorted(
            main_frames.items(),
            key=lambda item: _frame_sort_key(
                item[1], origin_point, corner_tolerance
            ),
            reverse=True,
        )
    )
    standard_frames = dict(itertools.islice(main_frames.items(), 0, max_frames))

    # Content-aware stamp/metadata selection (mirrors the POC decision tree).
    if len(standard_frames) > 1:
        vol1 = list(standard_frames.values())[0][1]
        vol2 = list(standard_frames.values())[1][1]
        if vol2 > vol1 * metadata_min_area_ratio:
            standard_frames.popitem()
            payload_bottom_y, _ = _find_payload_bottom_y(
                indexed_lines, standard_frames,
                inside_tolerance=int(payload_inside_tolerance),
                min_bottom_gap=payload_min_bottom_gap,
            )
            if payload_bottom_y is not None:
                stamp = _find_largest_frame_below_y(
                    main_frames, payload_bottom_y, inclusive=False
                )
                if stamp is not None:
                    stamp_key, (stamp_loop, stamp_square) = stamp
                    if stamp_square < vol1 * stamp_max_area_ratio:
                        standard_frames[stamp_key] = (stamp_loop, stamp_square)
                        standard_frames = dict(
                            itertools.islice(
                                sorted(
                                    standard_frames.items(),
                                    key=lambda item: item[1][1],
                                    reverse=True,
                                ),
                                0,
                                max_frames,
                            )
                        )
    if len(standard_frames) < max_frames:
        vol1 = (
            list(standard_frames.values())[0][1] if standard_frames else 0
        )
        stamp_top_y, _ = _find_stamp_top_y(
            indexed_lines, standard_frames,
            inside_tolerance=int(payload_inside_tolerance),
            max_length_ratio=stamp_max_length_ratio,
        )
        if stamp_top_y is not None:
            stamp = _find_largest_frame_below_y(
                main_frames, stamp_top_y, inclusive=False
            )
            if stamp is not None:
                stamp_key, (stamp_loop, stamp_square) = stamp
                if vol1 <= 0 or stamp_square < vol1 * stamp_max_area_ratio:
                    standard_frames[stamp_key] = (stamp_loop, stamp_square)
                    standard_frames = dict(
                        itertools.islice(
                            sorted(
                                standard_frames.items(),
                                key=lambda item: item[1][1],
                                reverse=True,
                            ),
                            0,
                            max_frames,
                        )
                    )

    frames: list[StampFrame] = []
    for loop, square in standard_frames.values():
        x1, y1, x2, y2 = _loop_bbox(loop)
        x2, y2 = origin_point if origin_point is not None else (x2, y2)
        frames.append(
            StampFrame(
                bbox=(int(x1), int(y1), int(x2), int(y2)), area=int(square)
            )
        )
    frames.sort(key=lambda f: f.area, reverse=True)
    return frames, corner_tolerance, deskewed, skew_angle, closed_loops, main_frames, indexed_lines, origin_point


# ---------------------------------------------------------------------------
# Frame splitting / region extraction
# ---------------------------------------------------------------------------


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    """Estimate the document background color.

    Samples the outer border stripe of the image (the area outside the central
    payload region) and returns the median RGB color, which for typical
    scanned documents is near-white.  Falls back to pure white if the stripe
    is empty.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    stripe_w = max(1, int(w * 0.1))
    stripe_h = max(1, int(h * 0.1))
    arr = np.array(rgb)
    top = arr[0:stripe_h, :, :].reshape(-1, 3)
    bottom = arr[h - stripe_h:h, :, :].reshape(-1, 3)
    left = arr[:, 0:stripe_w, :].reshape(-1, 3)
    right = arr[:, w - stripe_w:w, :].reshape(-1, 3)
    border = np.concatenate([top, bottom, left, right], axis=0)
    if border.size == 0:
        return (255, 255, 255)
    median = np.median(border, axis=0)
    return (int(median[0]), int(median[1]), int(median[2]))


def _erase_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    fill: tuple[int, int, int],
) -> Image.Image:
    """Return a copy of ``image`` with ``bbox`` filled with ``fill``."""
    out = image.copy()
    w, h = out.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 > x1 and y2 > y1:
        out.paste(fill, (x1, y1, x2, y2))
    return out


def _intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Intersection of two bboxes, or None if they do not overlap."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)
    return None


def split_frames(
    image: Image.Image,
) -> tuple[
    Image.Image | None,
    Image.Image,
    Image.Image | None,
    Image.Image | None,
    dict[str, StampFrame | None],
]:
    """Split a document page into four regions using the detected stamp frames.

    Uses [`find_stamp_frames()`](crown/stamp_frame.py) to locate up to two
    rectangular frames:

    * the **page content frame** -- the largest frame;
    * the **document metadata record frame** -- the second, smaller frame
      selected content-aware (the real stamp below the payload, when the naive
      second-largest frame is not it).

    All crops are taken from the (possibly deskewed) image returned by
    ``find_stamp_frames``, so the regions are consistent with the detected
    frame coordinates.

    Parameters
    ----------
    image:
        Source page as a PIL image.

    Returns
    -------
    tuple
        ``(metadata_interior, page_content, upper_right, bottom_right,
        frames_dict)`` where:

        * ``metadata_interior`` -- the interior of the metadata record frame,
          or ``None`` if the metadata frame was not found.
        * ``page_content`` -- the interior of the page content frame (or the
          whole deskewed image if no frames were found) with the metadata
          record frame erased, including its border (filled with the
          background color).
        * ``upper_right`` -- the upper-right corner of the page (a square of
          side = 10% of the page's lower dimension) with any overlapping area
          of the page content frame erased.
        * ``bottom_right`` -- the bottom-right corner of the page (same size)
          with any overlapping area of the page content frame erased.
        * ``frames_dict`` -- ``{"page_content_frame": ..., "metadata_frame":
          ...}`` with the detected :class:`StampFrame` objects (or ``None``).
    """
    rgb = image.convert("RGB")
    bg = _background_color(rgb)

    frames, corner_tolerance, deskewed, skew_angle, closed_loops, main_frames, indexed_lines, origin_point = find_stamp_frames(rgb, max_frames=2)
    work = deskewed
    w, h = work.size

    page_content_frame: StampFrame | None = None
    metadata_frame: StampFrame | None = None
    if frames:
        page_content_frame = frames[0]
        total_area = w * h
        content_area = page_content_frame.area
        area_ratio = content_area / total_area if total_area > 0 else 0
        if area_ratio < 0.7:
            page_content_frame = None
    if len(frames) >= 2 and page_content_frame:
        metadata_frame = frames[1]

    # 1. Interior of the metadata record frame.
    metadata_interior: Image.Image | None = None
    if metadata_frame is not None:
        x1, y1, x2, y2 = metadata_frame.bbox
        metadata_interior = work.crop(
            (max(0, x1 - 4), max(0, y1 - 4), min(w, x2 + 4), min(h, y2 + 4))
        )

    # 2. Page content interior with the metadata frame erased.
    if page_content_frame is not None:
        from crown.utils import bbox_expand
        dpi = int(image.info.get("dpi", (300, 300))[0])
        margin = max(w, h) // dpi
        x1, y1, x2, y2 = bbox_expand(page_content_frame.bbox, margin=margin, image_width=w, image_height=h)
        page_content = work.crop(
            (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        )
        if metadata_frame is not None:
            mx1, my1, mx2, my2 = metadata_frame.bbox
            local = (
                mx1 - x1,
                my1 - y1,
                mx2 - x1,
                my2 - y1,
            )
            page_content = _erase_region(page_content, local, bg)
    else:
        page_content = work.copy()
        if metadata_frame is not None:
            page_content = _erase_region(page_content, metadata_frame.bbox, bg)

    # Corner size: 10% of the page's lower dimension.
    corner_size = int(0.1 * min(w, h))

    def _corner(corner_bbox: tuple[int, int, int, int]) -> Image.Image:
        crop = work.crop(corner_bbox)
        tol = corner_tolerance // 2
        if page_content_frame is not None:
            overlap = _intersect(page_content_frame.bbox, corner_bbox)
            if overlap is not None:
                local = (
                    overlap[0] - corner_bbox[0],
                    overlap[1] - corner_bbox[1] - tol,
                    overlap[2] - corner_bbox[0] + tol,
                    overlap[3] - corner_bbox[1] + tol,
                )
                crop = _erase_region(crop, local, bg)
        bbox = crop.getbbox()
        if bbox is not None:
            bbox = (max(0, bbox[0] + tol), max(0, bbox[1] + tol),
                    min(crop.width, bbox[2] - tol), min(crop.height, bbox[3] - tol))
            crop = crop.crop(bbox)  # Remove any empty border.
        crop = ImageOps.expand(crop, border=corner_tolerance * 5, fill=bg)
        return crop

    # 3. Upper-right corner.
    upper_right = _corner((w - corner_size, 0, w, corner_size))

    # 4. Bottom-right corner.
    bottom_right = _corner((w - corner_size * 3, h - corner_size, w, h))

    frames_dict = {
        "page_content_frame": page_content_frame,
        "metadata_frame": metadata_frame,
    }
    metadata_interior = (
        ImageOps.expand(metadata_interior, border=corner_tolerance * 5, fill=bg)
        if metadata_interior
        else None
    )
    if crown_settings.DEBUG_FOLDER:
        os.makedirs(crown_settings.DEBUG_FOLDER, exist_ok=True)
        upper_right.save(f"{crown_settings.DEBUG_FOLDER}/upper_right.png")
        bottom_right.save(f"{crown_settings.DEBUG_FOLDER}/bottom_right.png")
    upper_right_text_boxes, upper_right = detect_texts(upper_right, tolerance=corner_tolerance)
    bottom_right_text_boxes, bottom_right = detect_texts(bottom_right, tolerance=corner_tolerance)
    return metadata_interior, page_content, upper_right, bottom_right, frames_dict
