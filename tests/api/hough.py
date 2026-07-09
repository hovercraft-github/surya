import sys
import cv2 as cv
import numpy as np
from random import randint
import itertools


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

def is_endpoint_outside_bbox(p1: tuple[int, int], p2: tuple[int, int], bbox: tuple[int, int, int, int], inclusive: bool = True) -> bool:
    """
    Checks if at least one of the line endpoints lies outside a bounding box.
    
    Parameters:
    p1 (tuple/list): Coordinates of the first endpoint (x1, y1)
    p2 (tuple/list): Coordinates of the second endpoint (x2, y2)
    bbox (tuple/list): Bounding box limits in the format (min_x, min_y, max_x, max_y)
    inclusive (bool): If True, points exactly on the boundary are considered INSIDE.
                      If False, points on the boundary are considered OUTSIDE.
    """
    x1, y1 = p1
    x2, y2 = p2
    min_x, min_y, max_x, max_y = bbox
    
    if inclusive:
        # Returns True if either point falls completely beyond the boundaries
        p1_outside = (x1 < min_x or x1 > max_x or y1 < min_y or y1 > max_y)
        p2_outside = (x2 < min_x or x2 > max_x or y2 < min_y or y2 > max_y)
    else:
        # Returns True if either point falls on or beyond the boundaries
        p1_outside = (x1 <= min_x or x1 >= max_x or y1 <= min_y or y1 >= max_y)
        p2_outside = (x2 <= min_x or x2 >= max_x or y2 <= min_y or y2 >= max_y)
        
    return p1_outside or p2_outside

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
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    # cdstP = np.copy(cdst)
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180 * 3, 200, None, 200, 10)
    raw_lines: list[list[int]] = [seg for seg in linesP.tolist()] if linesP is not None else []
    lines: list[list[int]] = deduplicate_segments(raw_lines, tolerance=15.0)
    # lines = sorted(lines, key=lambda l: math.hypot(l[2] - l[0], l[3] - l[1]), reverse=True)[:4]  # Keep only the longest 4 lines
    indexed_lines: dict[int, list[int]] = dict(enumerate(lines)) if lines else {}
    indexed_lines = {ix: seg for ix, seg in indexed_lines.items()
                     if is_endpoint_outside_bbox((seg[0], seg[1]), (seg[2], seg[3]), payload_bbox, inclusive=True)}
    origin_point = get_bottom_right_point(indexed_lines)
    intersections = find_all_segment_intersections(indexed_lines, tolerance=10.0)
    closed_loops = find_all_closed_loops(indexed_lines, intersections)
    print(f"Found {len(closed_loops)} closed loops (rectangles) in the image.")
    main_frames = {ix: seg for ix, seg in closed_loops.items() if path_has_points(seg[0], {origin_point}, tolerance=10)}
    standard_frames = dict(itertools.islice(sorted(main_frames.items(), key=lambda item: item[1][1], reverse=True), 0, 2))
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (randint(80,200),randint(80,200),randint(80,200)), 3, cv.LINE_AA)

    for val in standard_frames.values():
        loop, square = val
        x_coords = [point[1][0] for point in loop]
        y_coords = [point[1][1] for point in loop]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        cv.rectangle(cdstP, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4, cv.LINE_AA)

    cv.namedWindow("Scaled Window", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Scaled Window", 600, 800)
    # cv.imshow("Source", src)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    original_height, original_width = cdstP.shape[:2]
    new_width = 600
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)
    resized_image = cv.resize(cdstP, (new_width, new_height))
    cv.imshow("Probabilistic Line Transform", resized_image)
    
    cv.waitKey()
    cv.destroyAllWindows()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])
