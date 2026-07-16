"""Visual test driver for the production stamp-frame detector.

This script mirrors [`tests/api/hough.py`](tests/api/hough.py) but, instead of
re-implementing the Hough/loop pipeline inline, it delegates to the production
implementation in [`crown/stamp_frame.py`](crown/stamp_frame.py).

Usage:

    python tests/api/stamp_frame.py <image_file>

It loads the source page, runs [`find_stamp_frames()`](crown/stamp_frame.py:773)
(and [`split_frames()`](crown/stamp_frame.py:1123)) on it, then renders the
detected frames on top of the (possibly deskewed) image and opens a window so
the result can be inspected visually, just like the POC.
"""

import sys
import os

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu/"

import cv2 as cv
import numpy as np
from random import randint
from PIL import Image

# Make the project root importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crown.stamp_frame import find_stamp_frames, split_frames  # noqa: E402


def _pil_to_cv_rgb(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR MatLike."""
    arr = np.array(image.convert("RGB"))
    # PIL is RGB, OpenCV expects BGR.
    return cv.cvtColor(arr, cv.COLOR_RGB2BGR)


def main(argv):
    if len(argv) == 0:
        print("Error: no image file given!")
        print("Usage: stamp_frame.py <image_file>\n")
        return -1

    image_path = argv[0]
    try:
        src_pil = Image.open(image_path)
    except (FileNotFoundError, OSError) as exc:
        print(f"Error opening image: {exc}")
        print("Usage: stamp_frame.py <image_file>\n")
        return -1

    with Image.open(image_path) as img:
        dpi = img.info.get("dpi")
    print(f"DPI Resolution: {dpi}")

    # --- Production pipeline -------------------------------------------------
    frames, corner_tolerance, deskewed, skew_angle, closed_loops, main_frames, indexed_lines = find_stamp_frames(src_pil,
                                        max_frames=2,
                                        # skew_min_angle_deg=2.1
                                        )
    print(f"Detected {len(frames)} stamp frame(s); corner_tolerance={corner_tolerance}")
    for i, frame in enumerate(frames):
        x1, y1, x2, y2 = frame.bbox
        print(
            f"  frame[{i}]: bbox=({x1}, {y1})-({x2}, {y2}) "
            f"size={x2 - x1}x{y2 - y1} area={frame.area}"
        )

    # Also exercise split_frames() to make sure the full public API works.
    # metadata_interior, page_content, upper_right, bottom_right, frames_dict = split_frames(src_pil)
    # print(
    #     f"split_frames -> metadata_interior={'yes' if metadata_interior else 'no'}, "
    #     f"page_content={page_content.size}, upper_right={upper_right.size}, "
    #     f"bottom_right={bottom_right.size}"
    # )
    # print(
    #     f"  page_content_frame={frames_dict['page_content_frame']}, "
    #     f"metadata_frame={frames_dict['metadata_frame']}"
    # )

    # --- Visualization --------------------------------------------------------
    deskewed_cv = _pil_to_cv_rgb(deskewed)
    h_img, w_img = deskewed_cv.shape[:2]
    canvas = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    if indexed_lines:
        for i, l in indexed_lines.items():
            color = (randint(80,200),randint(80,200),randint(80,200))
            # if i in found_line_ix:
            #     color = (255, 0, 0)  # Highlight found lines in blue
            cv.line(canvas, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)

    # Draw every detected frame.
    colors = [
        (0, 0, 255),    # largest frame in red (BGR)
        (0, 255, 0),    # second frame in green
        (255, 0, 0),    # further frames in blue
    ]
    for i, frame in enumerate(frames):
        x1, y1, x2, y2 = frame.bbox
        color = colors[i % len(colors)]
        cv.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 4, cv.LINE_AA)
        label = f"frame{i}: {frame.area}"
        cv.putText(
            canvas, label, (int(x1) + 4, int(y1) + 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv.LINE_AA,
        )
    # if isinstance(closed_loops, dict):
    #     for frame in closed_loops.values():
    #         loop, area = frame
    #         x_coords = [point[1][0] for point in loop]
    #         y_coords = [point[1][1] for point in loop]
    #         x1, y1 = min(x_coords), min(y_coords)
    #         x2, y2 = max(x_coords), max(y_coords)
    #         color = (0, 255, 255)  # main frames in yellow
    #         cv.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 4, cv.LINE_AA)

    # Draw the metadata-interior crop (if any) in the top-left for reference.
    # if metadata_interior is not None:
    #     thumb = _pil_to_cv_rgb(metadata_interior)
    #     th = min(thumb.shape[0], h_img // 3)
    #     tw = int(thumb.shape[1] * th / max(thumb.shape[0], 1))
    #     thumb = cv.resize(thumb, (tw, th))
    #     canvas[0:th, 0:tw] = thumb
    #     cv.rectangle(canvas, (0, 0), (tw, th), (255, 255, 255), 2, cv.LINE_AA)
    #     cv.putText(
    #         canvas, "metadata_interior", (4, th + 22),
    #         cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA,
    #     )

    cv.namedWindow("Stamp Frames", cv.WINDOW_AUTOSIZE)
    new_height = 900
    aspect_ratio = new_height / h_img
    new_width = int(w_img * aspect_ratio)
    resized = cv.resize(canvas, (new_width, new_height))
    cv.imshow("Stamp Frames", resized)

    cv.waitKey()
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])