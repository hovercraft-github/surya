"""Detection of standard document stamp frames (rectangular border frames).

This module is the production implementation of the algorithm prototyped in
``tests/api/hough.py``.  It locates the rectangular frame lines that surround
the "payload" area of a scanned document page (the stamp frame) and returns the
two largest such frames.

The algorithm:

1. Convert the source image to grayscale and run Canny edge detection.
2. Run the probabilistic Hough line transform (``cv.HoughLinesP``).
3. Deduplicate near-identical line segments produced by Hough.
4. Keep only the segments whose endpoints lie in the outer border stripe
   (i.e. outside the central "payload" bbox) -- those are the frame lines.
5. Find all pairwise intersections of the surviving segments.
6. Walk the segment-intersection graph with DFS to discover closed rectangular
   loops (cycles of length 4 whose edges are long enough).
7. Keep the loops that touch the bottom-right corner of the frame and return
   the two largest by area.

All pixel-based tolerances (Hough parameters, dedup/intersection tolerances,
minimum edge length, corner-match tolerance) are scaled relative to the
reference page size of 2500x3500 pixels, for which the defaults were tuned.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps


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
# Geometry helpers (ported from the POC, unchanged in behaviour).
# ---------------------------------------------------------------------------


def _is_endpoint_outside_bbox(
    p1: tuple[int, int],
    p2: tuple[int, int],
    bbox: tuple[int, int, int, int],
    inclusive: bool = True,
) -> bool:
    """True if at least one endpoint lies outside ``bbox``.

    When ``inclusive`` is True, points exactly on the boundary count as inside.
    """
    x1, y1 = p1
    x2, y2 = p2
    min_x, min_y, max_x, max_y = bbox
    if inclusive:
        p1_outside = x1 < min_x or x1 > max_x or y1 < min_y or y1 > max_y
        p2_outside = x2 < min_x or x2 > max_x or y2 < min_y or y2 > max_y
    else:
        p1_outside = x1 <= min_x or x1 >= max_x or y1 <= min_y or y1 >= max_y
        p2_outside = x2 <= min_x or x2 >= max_x or y2 <= min_y or y2 >= max_y
    return p1_outside or p2_outside


def _get_segment_intersection(
    indexed_linesseg: dict[int, list[int]],
    ix1: int,
    ix2: int,
    tolerance: float = 5.0,
) -> tuple[frozenset[int], int, int] | None:
    """Intersection point of two segments, allowing ``tolerance`` pixels of
    overshoot beyond each segment's bounding box."""
    x1, y1, x2, y2 = indexed_linesseg[ix1]
    x3, y3, x4, y4 = indexed_linesseg[ix2]

    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-9:
        return None

    px = (b2 * c1 - b1 * c2) / determinant
    py = (a1 * c2 - a2 * c1) / determinant

    def is_on_segment(val, endpoint_a, endpoint_b, tol):
        return (min(endpoint_a, endpoint_b) - tol) <= val <= (
            max(endpoint_a, endpoint_b) + tol
        )

    if (
        is_on_segment(px, x1, x2, tolerance)
        and is_on_segment(py, y1, y2, tolerance)
        and is_on_segment(px, x3, x4, tolerance)
        and is_on_segment(py, y4, y3, tolerance)
    ):
        return (frozenset({ix1, ix2}), int(round(px)), int(round(py)))
    return None


def _find_all_segment_intersections(
    indexed_linesseg: dict[int, list[int]],
    tolerance: float = 5.0,
) -> dict[frozenset[int], tuple[int, int]]:
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
    return intersections


def _deduplicate_segments(
    segments: list[list[int]],
    tolerance: float = 5.0,
) -> list[list[int]]:
    """Collapse near-identical Hough segments into single averaged segments."""
    if not segments:
        return []

    def endpoints_close(a: list[int], b: list[int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        d1 = (ax1 - bx1) ** 2 + (ay1 - by1) ** 2
        d2 = (ax2 - bx2) ** 2 + (ay2 - by2) ** 2
        if d1 <= tolerance * tolerance and d2 <= tolerance * tolerance:
            return True
        d1 = (ax1 - bx2) ** 2 + (ay1 - by2) ** 2
        d2 = (ax2 - bx1) ** 2 + (ay2 - by1) ** 2
        return d1 <= tolerance * tolerance and d2 <= tolerance * tolerance

    def canonical(seg: list[int]) -> list[int]:
        x1, y1, x2, y2 = seg
        if (x1, y1) > (x2, y2):
            return [x2, y2, x1, y1]
        return [x1, y1, x2, y2]

    groups: list[list[list[int]]] = []
    for seg in segments:
        for group in groups:
            if endpoints_close(group[0], seg):
                group.append(seg)
                break
        else:
            groups.append([seg])

    averaged: list[list[int]] = []
    for group in groups:
        n = len(group)
        canon = [canonical(s) for s in group]
        sx1 = sum(s[0] for s in canon) / n
        sy1 = sum(s[1] for s in canon) / n
        sx2 = sum(s[2] for s in canon) / n
        sy2 = sum(s[3] for s in canon) / n
        averaged.append(
            [int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2))]
        )
    return averaged


def _edge_length(start_xy: tuple[int, int], end_xy: tuple[int, int]) -> int:
    x1, y1 = start_xy
    x2, y2 = end_xy
    return max(abs(x2 - x1), abs(y2 - y1))


def _get_bottom_right_point(
    indexed_linesseg: dict[int, list[int]]
) -> tuple[int, int]:
    point_x = 0
    point_y = 0
    for _, seg in indexed_linesseg.items():
        x1, y1, x2, y2 = seg
        if max(x1, x2) > point_x:
            point_x = max(x1, x2)
        if max(y1, y2) > point_y:
            point_y = max(y1, y2)
    return (point_x, point_y)


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
) -> dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]]:
    """Find all 4-edge rectangular loops in the segment-intersection graph."""
    adjacency: dict[int, dict[int, tuple[int, int]]] = {
        ix: dict() for ix in indexed_linesseg.keys()
    }
    for ix1, ix2 in intersections.keys():
        point = intersections[frozenset({ix1, ix2})]
        adjacency[ix1][ix2] = point
        adjacency[ix2][ix1] = point

    unique_loops: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]] = {}

    def dfs(
        current: int,
        start: int,
        visited: set[int],
        parent_path: list[tuple[int, tuple[int, int]]],
    ) -> None:
        if len(parent_path) >= 4:
            return
        visited.add(current)
        for neighbor, point in adjacency[current].items():
            path = parent_path.copy()
            path.append((current, point))
            if len(path) > 2:
                l1 = _edge_length(path[0][1], path[1][1])
                l2 = _edge_length(path[1][1], path[2][1])
                if l1 < edge_len_threshold or l2 < edge_len_threshold:
                    continue
            if neighbor == start and len(path) == 4:
                square = l1 * l2 if l1 and l2 else 0
                seg_set = frozenset(segment for segment, _ in path)
                if seg_set not in unique_loops:
                    unique_loops[seg_set] = (path, square)
            elif neighbor not in visited:
                dfs(neighbor, start, visited, path)
        visited.remove(current)

    for node in indexed_linesseg.keys():
        dfs(node, node, set(), [])

    return unique_loops


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
    hough_threshold: int = 200,
    hough_min_line_length: int = 200,
    hough_max_line_gap: int = 10,
    dedup_tolerance: float = 15.0,
    intersection_tolerance: float = 10.0,
    edge_len_threshold: int = 100,
    corner_tolerance: float = 10.0,
) -> tuple[list[StampFrame], int]:
    """Detect the standard rectangular stamp frames on a document page.

    The source is a PIL image (any mode); it is converted to grayscale
    internally.  All pixel-based tolerances are scaled relative to a reference
    page size of 2500x3500 pixels, for which the defaults were tuned, so the
    same defaults work across different scan resolutions.

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
    hough_threshold, hough_min_line_length, hough_max_line_gap:
        ``cv.HoughLinesP`` parameters.
    dedup_tolerance:
        Endpoint tolerance for collapsing duplicate Hough segments.
    intersection_tolerance:
        Tolerance (pixels) for accepting a segment-segment intersection.
    edge_len_threshold:
        Minimum edge length for a loop edge to be considered.
    corner_tolerance:
        Tolerance (pixels) for matching a loop corner to the bottom-right
        origin point of the frame.

    Returns
    -------
    list[StampFrame]
        Up to ``max_frames`` frames, sorted by area descending.
    """
    w_img, h_img = image.size

    # Scale all pixel-based parameters relative to the reference page.
    scale = ((w_img * h_img) / (_REF_W * _REF_H)) ** 0.5
    scale = max(scale, 1e-3)

    hough_threshold = max(1, int(round(hough_threshold * scale)))
    hough_min_line_length = max(1, int(round(hough_min_line_length * scale)))
    hough_max_line_gap = max(1, int(round(hough_max_line_gap * scale)))
    dedup_tolerance = dedup_tolerance * scale
    intersection_tolerance = intersection_tolerance * scale
    edge_len_threshold = max(1, int(round(edge_len_threshold * scale)))
    corner_tolerance = max(1, int(round(corner_tolerance * scale)))

    border_stripe_h = int(h_img * border_stripe_ratio)
    border_stripe_w = int(w_img * border_stripe_ratio)
    payload_bbox = (
        border_stripe_w,
        border_stripe_h,
        w_img - border_stripe_w,
        h_img - border_stripe_h,
    )

    # PIL -> OpenCV grayscale.
    gray = image.convert("L")
    src = np.array(gray)

    edges = cv.Canny(src, canny_low, canny_high, None, canny_aperture)

    lines_p = cv.HoughLinesP(
        edges,
        1,
        np.pi / 180 * 3,
        hough_threshold,
        None,
        hough_min_line_length,
        hough_max_line_gap,
    )
    raw_lines: list[list[int]] = (
        [seg for seg in lines_p.tolist()] if lines_p is not None else []
    )
    lines = _deduplicate_segments(raw_lines, tolerance=dedup_tolerance)

    indexed_lines: dict[int, list[int]] = (
        dict(enumerate(lines)) if lines else {}
    )
    indexed_lines = {
        ix: seg
        for ix, seg in indexed_lines.items()
        if _is_endpoint_outside_bbox(
            (seg[0], seg[1]), (seg[2], seg[3]), payload_bbox, inclusive=True
        )
    }
    if not indexed_lines:
        return [], corner_tolerance

    origin_point = _get_bottom_right_point(indexed_lines)
    intersections = _find_all_segment_intersections(
        indexed_lines, tolerance=intersection_tolerance
    )
    closed_loops = _find_all_closed_loops(
        indexed_lines, intersections, edge_len_threshold=edge_len_threshold
    )

    main_frames = {
        ix: val
        for ix, val in closed_loops.items()
        if _path_has_points(val[0], {origin_point}, tolerance=int(corner_tolerance))
    }

    ranked = sorted(
        main_frames.items(), key=lambda item: item[1][1], reverse=True
    )
    top = dict(itertools.islice(ranked, 0, max_frames))

    frames: list[StampFrame] = []
    for loop, square in top.values():
        x_coords = [point[1][0] for point in loop]
        y_coords = [point[1][1] for point in loop]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        frames.append(
            StampFrame(bbox=(int(x1), int(y1), int(x2), int(y2)), area=int(square))
        )
    return frames, corner_tolerance


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
) -> tuple[Image.Image | None, Image.Image, Image.Image, Image.Image, dict[str, StampFrame | None]]:
    """Split a document page into four regions using the detected stamp frames.

    Uses ``find_stamp_frames`` to locate up to two rectangular frames:

    * the **page content frame** -- the largest frame;
    * the **document metadata record frame** -- the second, smaller frame
      which always overlaps the page content frame.

    Parameters
    ----------
    image:
        Source page as a PIL image.

    Returns
    -------
    tuple
        ``(metadata_interior, page_content, upper_right, bottom_right)`` where:

        * ``metadata_interior`` -- the interior of the metadata record frame,
          or ``None`` if the metadata frame was not found.
        * ``page_content`` -- the interior of the page content frame (or the
          whole source image if no frames were found) with the metadata
          record frame erased, including its border (filled with the
          background color).
        * ``upper_right`` -- the upper-right corner of the page (a square of
          side = 10% of the page's lower dimension) with any overlapping area
          of the page content frame erased.
        * ``bottom_right`` -- the bottom-right corner of the page (same size)
          with any overlapping area of the page content frame erased.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    bg = _background_color(rgb)

    frames, corner_tolerance = find_stamp_frames(rgb, max_frames=2)

    page_content_frame: StampFrame | None = None
    metadata_frame: StampFrame | None = None
    if frames:
        page_content_frame = frames[0]
    if len(frames) >= 2:
        metadata_frame = frames[1]

    # 1. Interior of the metadata record frame.
    metadata_interior: Image.Image | None = None
    if metadata_frame is not None:
        x1, y1, x2, y2 = metadata_frame.bbox
        metadata_interior = rgb.crop(
            (max(0, x1-4), max(0, y1-4), min(w, x2+4), min(h, y2+4))
        )

    # 2. Page content interior with the metadata frame erased.
    if page_content_frame is not None:
        x1, y1, x2, y2 = page_content_frame.bbox
        page_content = rgb.crop(
            (max(0, x1+4), max(0, y1+4), min(w, x2-4), min(h, y2-4))
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
        page_content = rgb.copy()
        if metadata_frame is not None:
            page_content = _erase_region(page_content, metadata_frame.bbox, bg)

    # Corner size: 10% of the page's lower dimension.
    corner_size = int(0.1 * min(w, h))

    def _corner(corner_bbox: tuple[int, int, int, int]) -> Image.Image:
        crop = rgb.crop(corner_bbox)
        if page_content_frame is not None:
            overlap = _intersect(page_content_frame.bbox, corner_bbox)
            if overlap is not None:
                local = (
                    overlap[0] - corner_bbox[0],
                    overlap[1] - corner_bbox[1],
                    overlap[2] - corner_bbox[0] + corner_tolerance,
                    overlap[3] - corner_bbox[1] + corner_tolerance,
                )
                crop = _erase_region(crop, local, bg)
        crop = ImageOps.expand(crop, border=corner_tolerance*5, fill=bg)
        return crop

    # 3. Upper-right corner.
    upper_right = _corner((w - corner_size, 0, w, corner_size))

    # 4. Bottom-right corner.
    bottom_right = _corner((w - corner_size*3, h - corner_size, w, h))

    frames_dict = {
        "page_content_frame": page_content_frame,
        "metadata_frame": metadata_frame,
    }

    return metadata_interior, page_content, upper_right, bottom_right, frames_dict
