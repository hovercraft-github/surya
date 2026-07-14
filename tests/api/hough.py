import sys
import os
from turtle import right
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu/"
import cv2 as cv
import numpy as np
from random import randint
import itertools


def calc_point2seg_distance(px, py, x1, y1, x2, y2):
    """Calculates the distance from a point to a line segment."""
    # Vector AB
    ABx = x2 - x1
    ABy = y2 - y1
    # Vector AP
    APx = px - x1
    APy = py - y1
    is_hor = abs(ABy) < abs(ABx)  # Check if the segment is more horizontal than vertical

    # Dot products
    AB_AB = ABx * ABx + ABy * ABy  # |AB|^2
    if AB_AB == 0:
        return np.sqrt(APx * APx + APy * APy)  # A and B are the same point

    # Projection of AP onto AB, normalized by |AB|^2
    t = (APx * ABx + APy * ABy) / AB_AB

    proj = 'A'
    if t < 0.0:
        # Closest to A
        closest_x, closest_y = x1, y1
        proj = 'A'
        if is_hor == True:
            d_tang = abs(px - x1)
            d_ort = abs(py - y1)
        else:
            d_tang = abs(py - y1)
            d_ort = abs(px - x1)
    elif t > 1.0:
        # Closest to B
        closest_x, closest_y = x2, y2
        proj = 'B'
        if is_hor == True:
            d_tang = abs(px - x2)
            d_ort = abs(py - y2)
        else:
            d_tang = abs(py - y2)
            d_ort = abs(px - x2)
    else:
        # Projection falls on the segment
        closest_x = x1 + t * ABx
        closest_y = y1 + t * ABy
        proj = 'P'
        d_tang = 0
        d_ort = np.sqrt((closest_x - px) ** 2 + (closest_y - py) ** 2)

    return int(d_tang), int(d_ort), proj


def merge_collinear_lines(lines: np.ndarray, ort_tolerance=10, tang_tolerance=200, angle_tolerance_deg=2):
    """Merges overlapping, collinear line segments.

    lines: Numpy array of shape (N, 1, 4) or (N, 4) from cv2.HoughLinesP
    ort_tolerance: Max distance perpendicular to line & gap between
    segments
    angle_tolerance_deg: Max angular difference to consider
    collinear
    """
    if lines is None or len(lines) == 0:
        return []

    # Clean input shape from (N, 1, 4) to (N, 4)
    cleaned_lines = sorted(lines.reshape(-1, 4).tolist(), key=lambda x: (int(x[0]), int(x[1])))  # Sort by starting point
    min_length = tang_tolerance // 2
    cleaned_lines = [line for line in cleaned_lines if np.hypot(line[2] - line[0], line[3] - line[1]) >= min_length]
    # used: set[int] = set()
    # dropped: set[int] = set()
    absorbed = np.zeros(len(cleaned_lines), dtype=bool)

    for i, line1 in enumerate(cleaned_lines):
        if absorbed[i]:
            continue

        # used.add(i)
        x1, y1, x2, y2 = line1
        master_is_hor = abs(y2 - y1) < abs(x2 - x1)
        angle1 = 0 if master_is_hor else np.pi / 2

        for j, line2 in enumerate(cleaned_lines):
            if j == i:
                continue

            x3, y3, x4, y4 = line2
            slave_is_hor = abs(y4 - y3) < abs(x4 - x3)
            if slave_is_hor != master_is_hor:
                continue
            angle2 = np.arctan2(abs(y4 - y3), abs(x4 - x3))

            # 1. Check angle tolerance (handle wrap-around near pi)
            angle_diff = min(
                abs(angle1 - angle2),
                abs(angle1 - angle2 + np.pi),
                abs(angle1 - angle2 - np.pi),
            )

            if angle_diff > np.radians(angle_tolerance_deg):
                continue

            # Distance from (x3, y3) to line1
            d_tang1, d_ort1, proj1 = calc_point2seg_distance(x3, y3, x1, y1, x2, y2)
            a_within = d_ort1 <= ort_tolerance and d_tang1 <= tang_tolerance
            # Distance from (x4, y4) to line1
            d_tang2, d_ort2, proj2 = calc_point2seg_distance(x4, y4, x1, y1, x2, y2)
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
            line = []
            max_len = 0
            for c in candidates:
                l = np.hypot(c[2] - c[0], c[3] - c[1])
                if l > max_len:
                    max_len = l
                    line = c
            angle2 = np.arctan2(abs(line[3] - line[1]), abs(line[2] - line[0]))
            angle_diff = min(
                abs(angle1 - angle2),
                abs(angle1 - angle2 + np.pi),
                abs(angle1 - angle2 - np.pi),
            )
            if angle_diff > np.radians(angle_tolerance_deg):
                continue
            cleaned_lines[i] = line
            x1, y1, x2, y2 = line
            absorbed[j] = True
            # dropped.add(j)
            
    merged_lines = [line for i, line in enumerate(cleaned_lines) if not absorbed[i]]

    return merged_lines


def is_polygon_nested(poly_outer, poly_inner):
    """
    Checks if poly_inner is completely nested inside poly_outer.
    """
    # Iterate through all vertices of the inner polygon
    for point in poly_inner:
        # Measure distance from point to the outer polygon contour
        dist = cv.pointPolygonTest(poly_outer, tuple(point), measureDist=False)
        
        # If any point is outside (-1) or on the edge (0), it is not nested
        if dist < 1:
            return False
            
    return True

# --- Example Usage ---
# outer_poly = np.array([[10, 10], [10, 100], [100, 100], [100, 10], [10, 10]])
# inner_poly = np.array([[30, 30], [30, 70], [70, 70], [70, 30], [30, 30]])
# nested = is_polygon_nested(outer_poly, inner_poly)

def get_segment_intersection_(indexed_linesseg: dict[int, list[int]], ix1: int, ix2: int) -> tuple[frozenset[int], int, int] | None:
    """Finds the intersection point of two line segments if it exists."""
    seg1 = indexed_linesseg[ix1]
    seg2 = indexed_linesseg[ix2]
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2

    # Calculate denominator
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Lines are parallel or collinear

    # Check if the intersection point lies within both segments
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        # Calculate intersection coordinate
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (frozenset({ix1, ix2}), int(round(px)), int(round(py)))

    return None

def get_segment_intersection(indexed_linesseg: dict[int, list[int]], ix1: int, ix2: int, tolerance=5.0) -> tuple[frozenset[int], int, int] | None:
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

def find_all_segment_intersections(indexed_linesseg: dict[int, list[int]], tolerance=5.0) -> dict[frozenset[int], tuple[int, int]]:
    """Finds all intersection points between line segments."""
    intersections: dict[frozenset[int], tuple[int, int]] = {}
    for (ix1, ix2) in itertools.combinations(indexed_linesseg.keys(), 2):
        if frozenset({ix1, ix2}) in intersections:
            continue  # Skip if already found
        intersection = get_segment_intersection(indexed_linesseg, ix1, ix2, tolerance)
        if intersection is not None:
            key, x, y = intersection
            intersections[key] = (x, y)
    return intersections

def deduplicate_segments(
    segments: list[list[int]],
    tolerance: float = 5.0,
) -> list[list[int]]:
    """
    Detects duplicated line segments produced by cv.HoughLinesP and replaces
    each group of near-identical segments with a single average segment.

    Two segments are considered duplicates when both of their endpoints lie
    within ``tolerance`` pixels of each other. Endpoint ordering is ignored,
    so a segment ``[x1, y1, x2, y2]`` matches ``[x2, y2, x1, y1]`` as well.

    Parameters:
        segments: List of segments in ``[x1, y1, x2, y2]`` format (as returned
            by ``cv.HoughLinesP`` after flattening).
        tolerance: Maximum distance (in pixels) between corresponding
            endpoints for two segments to be treated as duplicates.

    Returns:
        A new list of averaged ``[x1, y1, x2, y2]`` segments with integer
        coordinates, preserving the order in which each duplicate group was
        first encountered.
    """
    if not segments:
        return []

    def endpoints_close(a: list[int], b: list[int]) -> bool:
        # Try both endpoint orderings so reversed segments also match.
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
        # Normalize endpoint order so that the lexicographically smaller
        # endpoint comes first. This prevents reversed duplicates from
        # cancelling each other out during averaging.
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
        averaged.append([int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2))])
    return averaged

def get_edge_length(start_xy: tuple[int, int], end_xy: tuple[int, int]) -> int:
    """Calculates the length of a line segment."""
    x1, y1 = start_xy
    x2, y2 = end_xy
    return max(abs(x2 - x1), abs(y2 - y1))

def get_seg_vertical_size(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculates the vertical size of a line segment."""
    return abs(y2 - y1)

def get_seg_horizontal_size(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculates the horizontal size of a line segment."""
    return abs(x2 - x1)

def get_seg_bottom_y(x1: int, y1: int, x2: int, y2: int) -> int:
    """Returns the bottom y-coordinate of a line segment."""
    return max(y1, y2)

def get_seg_top_y(x1: int, y1: int, x2: int, y2: int) -> int:
    """Returns the top y-coordinate of a line segment."""
    return min(y1, y2)

def get_seg_left_x(x1: int, y1: int, x2: int, y2: int) -> int:
    """Returns the left x-coordinate of a line segment."""
    return min(x1, x2)

def get_seg_right_x(x1: int, y1: int, x2: int, y2: int) -> int:
    """Returns the right x-coordinate of a line segment."""
    return max(x1, x2)

def is_endpoint_outside_bbox(p1: tuple[int, int], p2: tuple[int, int], bbox: tuple[int, int, int, int], tolerance: int = 10) -> bool:
    """
    Checks if at least one of the line endpoints lies outside a bounding box.
    
    Parameters:
    p1 (tuple[int, int]): Coordinates of the first endpoint (x1, y1)
    p2 (tuple[int, int]): Coordinates of the second endpoint (x2, y2)
    bbox (tuple[int, int, int, int]): Bounding box limits in the format (min_x, min_y, max_x, max_y)
    tolerance (int): The tolerance value for considering a point outside the bounding box.
    """
    x1, y1 = p1
    x2, y2 = p2
    min_x, min_y, max_x, max_y = bbox
    
    p1_outside = (x1 < min_x - tolerance or x1 > max_x + tolerance or y1 < min_y - tolerance or y1 > max_y + tolerance)
    p2_outside = (x2 < min_x - tolerance or x2 > max_x + tolerance or y2 < min_y - tolerance or y2 > max_y + tolerance)
        
    return p1_outside or p2_outside

def is_endpoint_inside_bbox(p1: tuple[int, int], p2: tuple[int, int], bbox: tuple[int, int, int, int], tolerance: int = 10) -> bool:
    """
    Checks if at least one of the line endpoints lies inside a bounding box.
    
    Parameters:
    p1 (tuple[int, int]): Coordinates of the first endpoint (x1, y1)
    p2 (tuple[int, int]): Coordinates of the second endpoint (x2, y2)
    bbox (tuple[int, int, int, int]): Bounding box limits in the format (min_x, min_y, max_x, max_y)
    tolerance (int): The tolerance value for considering a point inside the bounding box.
    """
    x1, y1 = p1
    x2, y2 = p2
    min_x, min_y, max_x, max_y = bbox
    
    # Returns True if either point falls within the boundaries considering the tolerance
    p1_inside = (x1 >= min_x - tolerance and x1 <= max_x + tolerance and y1 >= min_y - tolerance and y1 <= max_y + tolerance)
    p2_inside = (x2 >= min_x - tolerance and x2 <= max_x + tolerance and y2 >= min_y - tolerance and y2 <= max_y + tolerance)
        
    return p1_inside or p2_inside

def get_bottom_right_point(indexed_linesseg: dict[int, list[int]]) -> tuple[int, int]:
    """Returns the bottom-right point of each line segment."""
    point_x: int = 0
    point_y: int = 0
    for ix, seg in indexed_linesseg.items():
        x1, y1, x2, y2 = seg
        if max(x1, x2) > point_x:
            point_x = max(x1, x2)
        if max(y1, y2) > point_y:
            point_y = max(y1, y2)
    return (point_x, point_y)

def find_bottom_right_corner_of_loops(
    closed_loops: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]],
) -> tuple[int, int] | None:
    """Returns the coordinates of the most bottom-right corner among all
    loops in ``closed_loops``.

    Each loop is a list of ``(segment_index, (x, y))`` corner points (as
    produced by [`find_all_closed_loops()`](tests/api/hough.py:412)). The
    "most bottom-right" corner is the one with the greatest ``x + y`` sum;
    ties are broken by the greatest ``x`` and then the greatest ``y``.

    Parameters:
        closed_loops: Output of ``find_all_closed_loops``. Each value is a
            ``(loop, square)`` tuple where ``loop`` is a list of
            ``(segment_index, (x, y))`` points.

    Returns:
        The ``(x, y)`` coordinates of the most bottom-right corner, or
        ``None`` if ``closed_loops`` is empty or contains no corner points.
    """
    best_point: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for loop, _ in closed_loops.values():
        for _, (x, y) in loop:
            key = (x + y, x, y)
            if best_key is None or key > best_key:
                best_key = key
                best_point = (x, y)
    return best_point

def path_has_points(path: list[tuple[int, tuple[int, int]]], points: set[tuple[int, int]], tolerance: int) -> bool:
    """Checks if any point in the path is in the given set of points."""
    for _, point in path:
        for px, py in points:
            if abs(point[0] - px) <= tolerance and abs(point[1] - py) <= tolerance:
                return True
    return False

def find_all_closed_loops(
    indexed_linesseg: dict[int, list[int]],
    intersections: dict[frozenset[int], tuple[int, int]],
    edge_len_threshold=100,
    square_threshold: int | None = None
) -> dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]]:
    """Finds all closed loops formed by the line segments."""
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

    return unique_loops


def find_payload_bottom_y(
    indexed_lines: dict[int, list[int]],
    standard_frames: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]],
    inside_tolerance: int = 10,
    min_bottom_gap: int = 200,
) -> tuple[int | None, list[int]]:
    """Finds the bottom y-coordinate among the vertical lines
    contained inside the first standard frame whose upper point is above the half of
    the frame height.

    A line is considered vertical when its vertical size is strictly greater
    than its horizontal size (see ``get_seg_vertical_size`` /
    ``get_seg_horizontal_size``). A line is considered inside the frame when
    both of its endpoints fall within the frame bounding box (optionally
    expanded by ``inside_tolerance`` pixels). The line length is the euclidean
    distance between its endpoints.

    Parameters:
        indexed_lines: Mapping of line index to ``[x1, y1, x2, y2]`` segments.
        standard_frames: Output of ``find_all_closed_loops`` filtered down to
            the standard frames. Only its first item (insertion order) is used.
        inside_tolerance: Extra margin (in pixels) added to the frame bbox
            when testing whether a line is inside the frame.
        min_bottom_gap: Minimum distance (in pixels) from the bottom of the frame to the vertical line.

    Returns:
        The bottom-most y-coordinate plus inside_tolerance as an int, or ``None`` if there
        is no such vertical line / no standard frame.
    """
    if not standard_frames:
        return None, []

    loop, _ = next(iter(standard_frames.values()))
    x_coords = [point[1][0] for point in loop]
    y_coords = [point[1][1] for point in loop]
    frame_x1, frame_y1 = min(x_coords), min(y_coords)
    frame_x2, frame_y2 = max(x_coords), max(y_coords)
    frame_height = frame_y2 - frame_y1
    if frame_height <= 0:
        return None, []
    min_length = frame_height / 10

    frame_v_borders = []
    prev_point = loop[-1][1]
    for item in loop:
        index, (x, y) = item
        x1, y1 = prev_point
        x2, y2 = x, y
        if get_seg_vertical_size(x1, y1, x2, y2) > get_seg_horizontal_size(x1, y1, x2, y2):
            frame_v_borders.append(index)
        prev_point = (x, y)
    # print("Main frame vertical borders:")
    # for index in frame_v_borders:
    #     if index not in indexed_lines:
    #         print(f"  Index {index}: Not found in indexed_lines")
    #     else:
    #         x1, y1, x2, y2 = indexed_lines[index]
    #         print(f"  Index {index}: ({x1}, {y1}) -> ({x2}, {y2})")
    segment_indexes: list[int] = []
    bottom_ys: list[int] = []
    for index, seg in indexed_lines.items():
        if index in frame_v_borders:
            continue
        x1, y1, x2, y2 = seg
        if abs(max(y1, y2) - frame_y2) <= min_bottom_gap or max(y1, y2) >= frame_y2:
            continue
        # Only vertical segments are of interest.
        if get_seg_vertical_size(x1, y1, x2, y2) <= get_seg_horizontal_size(x1, y1, x2, y2):
            continue
        # Both endpoints must lie inside the frame bounding box.
        if is_endpoint_outside_bbox(
            (x1, y1), (x2, y2),
            (frame_x1, frame_y1, frame_x2, frame_y2),
            tolerance=inside_tolerance,
        ):
            continue
        if not is_endpoint_inside_bbox(
            (x1, y1), (x2, y2),
            (frame_x1, frame_y1, frame_x2, frame_y2 // 2),
            tolerance=inside_tolerance,
        ):
            continue
        # Length must be greater than half of the frame height.
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length <= min_length:
            continue
        segment_indexes.append(index)
        bottom_ys.append(get_seg_bottom_y(x1, y1, x2, y2))

    if not bottom_ys:
        return None, []

    # print("Vertical segments:")
    # for index in segment_indexes:
    #     x1, y1, x2, y2 = indexed_lines[index]
    #     print(f"  Index {index}: ({x1}, {y1}) -> ({x2}, {y2})")
    bottom_ys = sorted(bottom_ys, reverse=True)
    return bottom_ys[0] + inside_tolerance, segment_indexes


def find_stamp_top_y(
    indexed_lines: dict[int, list[int]],
    standard_frames: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]],
    inside_tolerance: int = 10,
    max_length_ratio: float = 0.25,
) -> tuple[int | None, list[int]]:
    """Finds the top y-coordinate among the upper ends of the vertical lines
    touching the bottom line of the first standard frame and having length
    less than ``max_length_ratio`` (default 1/4) of the frame height.

    A line is considered vertical when its vertical size is strictly greater
    than its horizontal size (see ``get_seg_vertical_size`` /
    ``get_seg_horizontal_size``). A line is considered inside the frame when
    both of its endpoints fall within the frame bounding box (optionally
    expanded by ``inside_tolerance`` pixels). A line is considered to touch
    the bottom line of the frame when its bottom y-coordinate is within
    ``inside_tolerance`` pixels of the frame bottom. The line length is the
    euclidean distance between its endpoints.

    Parameters:
        indexed_lines: Mapping of line index to ``[x1, y1, x2, y2]`` segments.
        standard_frames: Output of ``find_all_closed_loops`` filtered down to
            the standard frames. Only its first item (insertion order) is used.
        inside_tolerance: Extra margin (in pixels) added to the frame bbox
            when testing whether a line is inside the frame, and also used as
            the tolerance when checking that the line touches the frame bottom.
        max_length_ratio: Maximum allowed line length expressed as a fraction
            of the frame height (defaults to 0.25, i.e. 1/4).

    Returns:
        The top-most y-coordinate minus inside_tolerance as an int, or ``None``
        if there is no such vertical line / no standard frame.
    """
    if not standard_frames:
        return None, []

    loop, _ = next(iter(standard_frames.values()))
    x_coords = [point[1][0] for point in loop]
    y_coords = [point[1][1] for point in loop]
    frame_x1, frame_y1 = min(x_coords), min(y_coords)
    frame_x2, frame_y2 = max(x_coords), max(y_coords)
    frame_height = frame_y2 - frame_y1
    if frame_height <= 0:
        return None, []
    max_length = frame_height * max_length_ratio

    frame_v_borders = []
    prev_point = loop[-1][1]
    for item in loop:
        index, (x, y) = item
        x1, y1 = prev_point
        x2, y2 = x, y
        if get_seg_vertical_size(x1, y1, x2, y2) > get_seg_horizontal_size(x1, y1, x2, y2):
            frame_v_borders.append(index)
        prev_point = (x, y)

    segment_indexes: list[int] = []
    top_ys: list[int] = []
    for index, seg in indexed_lines.items():
        if index in frame_v_borders:
            continue
        x1, y1, x2, y2 = seg
        # Only vertical segments are of interest.
        if get_seg_vertical_size(x1, y1, x2, y2) <= get_seg_horizontal_size(x1, y1, x2, y2):
            continue
        # Both endpoints must lie inside the frame bounding box.
        if is_endpoint_outside_bbox(
            (x1, y1), (x2, y2),
            (frame_x1, frame_y1, frame_x2, frame_y2),
            tolerance=inside_tolerance,
        ):
            continue
        # The line must touch the bottom line of the frame.
        if abs(get_seg_bottom_y(x1, y1, x2, y2) - frame_y2) > inside_tolerance:
            continue
        # Length must be less than max_length_ratio of the frame height.
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= max_length:
            continue
        segment_indexes.append(index)
        top_ys.append(get_seg_top_y(x1, y1, x2, y2))

    if not top_ys:
        return None, []

    top_ys = sorted(top_ys)
    return top_ys[0] - inside_tolerance, segment_indexes


def find_largest_frame_below_y(
    main_frames: dict[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]],
    y_threshold: int,
    inclusive: bool = False,
) -> tuple[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]] | None:
    """Returns the biggest (by surface area) frame among ``main_frames`` that
    lies entirely below the specified y position.

    A frame is considered to lie entirely below ``y_threshold`` when its
    topmost y-coordinate (the minimum y of its loop corner points) is not
    less than ``y_threshold``. With ``inclusive=True`` (the default) a frame
    whose top edge sits exactly on ``y_threshold`` also qualifies; with
    ``inclusive=False`` it must be strictly below.

    Parameters:
        main_frames: Output of ``find_all_closed_loops`` filtered down to the
            main frames. Each value is a ``(loop, square)`` tuple, where
            ``loop`` is a list of ``(segment_index, (x, y))`` points and
            ``square`` is the precomputed surface area.
        y_threshold: The y position the frame must lie below.
        inclusive: If True, a frame whose top edge equals ``y_threshold``
            qualifies; if False, it must be strictly greater.

    Returns:
        The ``(key, (loop, square))`` entry of the largest qualifying frame,
        or ``None`` if no frame lies entirely below ``y_threshold``.
    """
    best_entry: tuple[frozenset, tuple[list[tuple[int, tuple[int, int]]], int]] | None = None
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


def calc_average_skew_angle(
    lines: list[list],
    max_count: int = 20,
    vertical_tolerance_deg: float = 20.0,
) -> tuple[float, list[list[int]]]:
    """Selects the ``max_count`` largest almost-vertical line segments found by
    ``cv2.HoughLinesP`` and returns the average skew angle of the image.

    A line is considered *almost vertical* when the absolute angle between the
    segment and the vertical axis does not exceed ``vertical_tolerance_deg``
    degrees. The skew angle of a single line is the signed deviation (in
    degrees) of that line from the true vertical, i.e. the angle of the
    segment measured from the y-axis, positive when the line leans to the
    right and negative when it leans to the left. The average of these signed
    deviations across the selected lines is returned as the image skew angle,
    which can be used directly with ``cv2.getRotationMatrix2D`` to deskew the
    image.

    Parameters:
        lines: Array returned by ``cv2.HoughLinesP`` of shape ``(N, 1, 4)`` or
            ``(N, 4)``. ``None`` / empty input is handled gracefully.
        max_count: Maximum number of the longest almost-vertical lines to use
            when computing the average (default ``20``).
        vertical_tolerance_deg: Maximum absolute deviation from the vertical
            axis (in degrees) for a line to be considered almost vertical
            (default ``20.0``).

    Returns:
        A ``(skew_angle_deg, selected_lines)`` tuple where ``skew_angle_deg``
        is the average signed skew angle in degrees (``0.0`` when no suitable
        lines are found) and ``selected_lines`` is the list of the
        ``[x1, y1, x2, y2]`` segments that were used for the calculation,
        sorted by descending length.
    """
    if lines is None:
        return 0.0, []

    # Normalize the shape produced by cv2.HoughLinesP to (N, 4).
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
        # Angle of the segment measured from the vertical (y) axis.
        # atan2(dx, dy) yields 0 for a perfectly vertical line.
        angle_from_vertical = float(np.arctan2(dx, dy))
        # A line and its 180-degree rotation describe the same physical line,
        # so fold the angle into [-pi/2, pi/2].
        if angle_from_vertical > np.pi / 2:
            angle_from_vertical -= np.pi
        elif angle_from_vertical < -np.pi / 2:
            angle_from_vertical += np.pi
        if abs(angle_from_vertical) > tol_rad:
            continue
        candidates.append((length, angle_from_vertical, [x1, y1, x2, y2]))

    if not candidates:
        return 0.0, []

    # Pick the longest almost-vertical lines.
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_count]

    skew_angle_deg = float(np.degrees(np.mean([a for _, a, _ in selected])))
    selected_lines = [seg for _, _, seg in selected]
    return skew_angle_deg, selected_lines


def calc_most_probable_skew_angle(
    lines: list[list],
    max_count: int = 20,
    vertical_tolerance_deg: float = 20.0,
    bin_size_deg: float = 0.5,
) -> tuple[float, list[list[int]]]:
    """Selects the ``max_count`` largest almost-vertical line segments found by
    ``cv2.HoughLinesP`` and returns the *most probable* skew angle of the
    image.

    Unlike [`calc_average_skew_angle()`](tests/api/hough.py:597), which simply
    averages the signed deviations of the selected lines, this function
    estimates the mode of the angle distribution. Each selected line is
    weighted by its length (longer lines are more reliable witnesses of the
    true document orientation) and accumulated into angular bins of width
    ``bin_size_deg``. The center of the bin with the largest total weight is
    returned as the most probable skew angle. This is more robust than the
    plain average against a few grossly misdetected outliers, because a stray
    short line contributes little weight and cannot pull the mode away from the
    dominant cluster.

    A line is considered *almost vertical* when the absolute angle between the
    segment and the vertical axis does not exceed ``vertical_tolerance_deg``
    degrees. The skew angle of a single line is the signed deviation (in
    degrees) of that line from the true vertical, i.e. the angle of the
    segment measured from the y-axis, positive when the line leans to the
    right and negative when it leans to the left. The returned angle can be
    used directly with ``cv2.getRotationMatrix2D`` to deskew the image.

    Parameters:
        lines: Array returned by ``cv2.HoughLinesP`` of shape ``(N, 1, 4)`` or
            ``(N, 4)``. ``None`` / empty input is handled gracefully.
        max_count: Maximum number of the longest almost-vertical lines to use
            when computing the angle (default ``20``).
        vertical_tolerance_deg: Maximum absolute deviation from the vertical
            axis (in degrees) for a line to be considered almost vertical
            (default ``20.0``).
        bin_size_deg: Width (in degrees) of the angular bins used to estimate
            the mode of the angle distribution (default ``0.5``). Smaller
            values give finer resolution but need more lines to be stable.

    Returns:
        A ``(skew_angle_deg, selected_lines)`` tuple where ``skew_angle_deg``
        is the most probable signed skew angle in degrees (``0.0`` when no
        suitable lines are found) and ``selected_lines`` is the list of the
        ``[x1, y1, x2, y2]`` segments that were used for the calculation,
        sorted by descending length.
    """
    if lines is None:
        return 0.0, []

    # Normalize the shape produced by cv2.HoughLinesP to (N, 4).
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
        # Angle of the segment measured from the vertical (y) axis.
        # atan2(dx, dy) yields 0 for a perfectly vertical line.
        angle_from_vertical = float(np.arctan2(dx, dy))
        # A line and its 180-degree rotation describe the same physical line,
        # so fold the angle into [-pi/2, pi/2].
        if angle_from_vertical > np.pi / 2:
            angle_from_vertical -= np.pi
        elif angle_from_vertical < -np.pi / 2:
            angle_from_vertical += np.pi
        if abs(angle_from_vertical) > tol_rad:
            continue
        candidates.append((length, angle_from_vertical, [x1, y1, x2, y2]))

    if not candidates:
        return 0.0, []

    # Pick the longest almost-vertical lines.
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_count]

    # Length-weighted histogram to estimate the mode of the angle
    # distribution. Longer lines are more reliable, so they weigh more.
    angles_deg = np.degrees([a for _, a, _ in selected])
    weights = np.array([l for l, _, _ in selected], dtype=float)

    if bin_size_deg <= 0:
        bin_size_deg = 0.5
    # Bin edges centered on multiples of bin_size_deg so the mode is the
    # center of the winning bin rather than an arbitrary edge.
    min_angle = float(np.min(angles_deg))
    max_angle = float(np.max(angles_deg))
    # Ensure at least one bin even when all angles are identical.
    span = max(max_angle - min_angle, bin_size_deg)
    # Extend the range by half a bin on each side so edge angles fall inside.
    left = min_angle - bin_size_deg / 2
    right = max_angle + bin_size_deg / 2
    n_bins = max(int(np.ceil((right - left) / bin_size_deg)), 1)
    edges = np.linspace(left, left + n_bins * bin_size_deg, n_bins + 1)

    hist, edges = np.histogram(angles_deg, bins=edges, weights=weights)
    best_bin = int(np.argmax(hist))
    # Center of the winning bin is the most probable angle.
    skew_angle_deg = float((edges[best_bin] + edges[best_bin + 1]) / 2.0)

    selected_lines = [seg for _, _, seg in selected]
    return skew_angle_deg, selected_lines


def main(argv):
    
    default_file = 'sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    h_img, w_img = src.shape[:2]
    border_stripe_h = int(h_img * 0.1)  # 10% of image height
    border_stripe_w = int(w_img * 0.1)  # 10% of image width
    payload_bbox = (border_stripe_w, border_stripe_h, w_img - border_stripe_w, h_img - border_stripe_h)
    
    dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    # cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    # cdstP = np.copy(cdst)
    cv.bitwise_not(src, dst)
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180 * 1, 100, None, 100, 20)
    lines = merge_collinear_lines(linesP, ort_tolerance=20, tang_tolerance=100, angle_tolerance_deg=5)
    skew_angle, _ = calc_most_probable_skew_angle(lines, vertical_tolerance_deg=5.0)
    if abs(skew_angle) > 0.1:
        print(f"Detected skew angle: {skew_angle:.2f} degrees. Deskewing the image...")
        center = (w_img // 2, h_img // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, skew_angle * -1, 1.0)
        src = cv.warpAffine(src, rotation_matrix, (w_img, h_img), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE) #, INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        dst = cv.Canny(src, 50, 200, None, 3)
        cdstP = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        cv.bitwise_not(src, dst)
        linesP = cv.HoughLinesP(dst, 2, np.pi / 180 * 2, 100, None, 100, 20)
        lines = merge_collinear_lines(linesP, ort_tolerance=20, tang_tolerance=100, angle_tolerance_deg=5)
    raw_lines: list[list[int]] = [seg for seg in linesP.tolist()] if linesP is not None else []
    # lines: list[list[int]] = deduplicate_segments(raw_lines, tolerance=30.0)
    lines: list[list[int]] = deduplicate_segments(lines, tolerance=30.0)
    print(f"Found {len(lines)} unique lines in the image.")
    # lines = sorted(lines, key=lambda l: math.hypot(l[2] - l[0], l[3] - l[1]), reverse=True)[:4]  # Keep only the longest 4 lines
    indexed_lines: dict[int, list[int]] = dict(enumerate(lines)) if lines else {}
    outer_lines = {ix: seg for ix, seg in indexed_lines.items()
                     if is_endpoint_outside_bbox((seg[0], seg[1]), (seg[2], seg[3]), payload_bbox, tolerance=1)}
    # origin_point = get_bottom_right_point(outer_lines)
    intersections = find_all_segment_intersections(outer_lines, tolerance=20.0)
    closed_loops = find_all_closed_loops(outer_lines, intersections, edge_len_threshold=100, square_threshold=200000)
    origin_point = find_bottom_right_corner_of_loops(closed_loops)
    print(f"Found {len(closed_loops)} closed loops (rectangles) in the image.")
    origin_points = set([origin_point]) if origin_point is not None else set()
    main_frames = {ix: seg for ix, seg in closed_loops.items() if path_has_points(seg[0], origin_points, tolerance=100)}
    main_frames = dict(sorted(main_frames.items(), key=lambda item: item[1][1], reverse=True))
    standard_frames = dict(itertools.islice(main_frames.items(), 0, 2))
    found_line_ix: list[int] = []
    payload_bottom_y: int | None = None
    if len(standard_frames) > 1:
        vol1 = list(standard_frames.values())[0][1]
        vol2 = list(standard_frames.values())[1][1]
        if vol2 > vol1//5:
            standard_frames.popitem()
            payload_bottom_y, found_line_ix = find_payload_bottom_y(indexed_lines, standard_frames, inside_tolerance=10, min_bottom_gap=200)
            if payload_bottom_y is not None:
                print(f"Payload bottom y-coordinate: {payload_bottom_y}")
                stamp = find_largest_frame_below_y(main_frames, payload_bottom_y, inclusive=False)
                if stamp is not None:
                    stamp_key, (stamp_loop, stamp_square) = stamp
                    l1 = get_edge_length(stamp_loop[0][1], stamp_loop[1][1])
                    l2 = get_edge_length(stamp_loop[1][1], stamp_loop[2][1])
                    l3 = get_edge_length(stamp_loop[2][1], stamp_loop[3][1])
                    l4 = get_edge_length(stamp_loop[3][1], stamp_loop[0][1])
                    print(f"Found stamp frame with edges: {l1}, {l2}, {l3}, {l4} and area: {stamp_square}")
                    if stamp_square < vol1//4:
                        standard_frames[stamp_key] = (stamp_loop, stamp_square)
                        standard_frames = dict(itertools.islice(sorted(standard_frames.items(), key=lambda item: item[1][1], reverse=True), 0, 2))
        if len(standard_frames) < 2:
            payload_bottom_y, found_line_ix = find_stamp_top_y(indexed_lines, standard_frames, inside_tolerance=10)
            if payload_bottom_y is not None:
                print(f"Stamp top y-coordinate: {payload_bottom_y}")
                stamp = find_largest_frame_below_y(main_frames, payload_bottom_y, inclusive=False)
                if stamp is not None:
                    stamp_key, (stamp_loop, stamp_square) = stamp
                    l1 = get_edge_length(stamp_loop[0][1], stamp_loop[1][1])
                    l2 = get_edge_length(stamp_loop[1][1], stamp_loop[2][1])
                    l3 = get_edge_length(stamp_loop[2][1], stamp_loop[3][1])
                    l4 = get_edge_length(stamp_loop[3][1], stamp_loop[0][1])
                    print(f"Found stamp frame with edges: {l1}, {l2}, {l3}, {l4} and area: {stamp_square}")
                    if stamp_square < vol1//4:
                        standard_frames[stamp_key] = (stamp_loop, stamp_square)
                        standard_frames = dict(itertools.islice(sorted(standard_frames.items(), key=lambda item: item[1][1], reverse=True), 0, 2))
    
    # if raw_lines:
    #     for l in raw_lines:
    #         color = (randint(80,200),randint(80,200),randint(80,200))
    #         cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)

    if indexed_lines:
        for i, l in indexed_lines.items():
            color = (randint(80,200),randint(80,200),randint(80,200))
            if i in found_line_ix:
                color = (255, 0, 0)  # Highlight found lines in blue
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)

    if origin_point:
        for val in standard_frames.values():
            loop, square = val
            x_coords = [point[1][0] for point in loop]
            y_coords = [point[1][1] for point in loop]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = origin_point #max(x_coords), max(y_coords)
            cv.rectangle(cdstP, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4, cv.LINE_AA)

    cv.namedWindow("Scaled Window", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Scaled Window", 600, 800)
    # cv.imshow("Source", src)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    original_height, original_width = cdstP.shape[:2]
    new_height = 900
    aspect_ratio = new_height / original_height
    new_width = int(original_width * aspect_ratio)
    resized_image = cv.resize(cdstP, (new_width, new_height))
    cv.imshow("Probabilistic Line Transform", resized_image)
    
    cv.waitKey()
    cv.destroyAllWindows()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])
