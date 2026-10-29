"""
Physical layout of the RustReader measuring mat (printed on A4 paper).

All values are in millimetres, measured from the top-left corner of the page.
Four ArUco markers sit in the corners. Because we know exactly where each
marker corner is on paper, we can undo the phone's perspective and convert
pixels into millimetres.

If you change anything here, re-run tools/make_marker_sheet.py and re-print.
"""
import cv2

PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 297.0

ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50
MARKER_SIDE_MM = 30.0

# Top-left corner (x, y) of each marker on the page, keyed by marker id.
MARKER_TOP_LEFT_MM = {
    0: (15.0, 15.0),     # top-left
    1: (165.0, 15.0),    # top-right
    2: (165.0, 252.0),   # bottom-right
    3: (15.0, 252.0),    # bottom-left
}

# The white area where the user places ONE bolt (x_min, y_min, x_max, y_max).
WORK_AREA_MM = (15.0, 52.0, 195.0, 245.0)

# Resolution of the straightened ("rectified") top-down image we build.
PIXELS_PER_MM = 12.0


def marker_corners_on_paper_mm(marker_id):
    """Return the 4 corners of a marker in mm, in ArUco order:
    top-left, top-right, bottom-right, bottom-left."""
    left, top = MARKER_TOP_LEFT_MM[marker_id]
    right = left + MARKER_SIDE_MM
    bottom = top + MARKER_SIDE_MM
    return [(left, top), (right, top), (right, bottom), (left, bottom)]
