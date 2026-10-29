"""
Finds the fastener(s) lying on the white work area of the straightened mat.

Rusted steel is either darker than the paper or clearly orange/brown, so an
object pixel is anything noticeably darker than the local paper colour OR
strongly coloured. Comparing against the LOCAL paper colour (not one fixed
threshold) keeps uneven lighting and soft shadows from fooling us.
"""
import cv2
import numpy as np

from vision import mat_layout

DARKER_THAN_PAPER_BY = 45          # grey levels, minimum
SHADOW_REJECT_FRACTION = 0.52      # must be darker than paper by this share of paper brightness;
                                   # soft shadows only dim paper by ~20-40%, steel by 60%+
MINIMUM_RUST_SATURATION = 70       # HSV saturation (0-255)
MINIMUM_OBJECT_AREA_MM2 = 15.0
EDGE_GROW_MM = 0.8
BORDER_INSET_MM = 3.0              # skip the dashed outline printed on the mat


def estimate_paper_background(gray_image):
    """Brightness the paper WOULD have at every pixel, ignoring small objects.
    A large median filter on a shrunken copy removes bolts but keeps lighting."""
    shrink_factor = 8
    small = cv2.resize(gray_image, None, fx=1 / shrink_factor, fy=1 / shrink_factor,
                       interpolation=cv2.INTER_AREA)
    small_background = cv2.medianBlur(small, 51)
    return cv2.resize(small_background, (gray_image.shape[1], gray_image.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def grow_core_to_true_edge(core_mask, gray_image, paper_background):
    """The strict core misses the blurred rim of the bolt. Within EDGE_GROW_MM of the
    core, also accept pixels darker than half-way between bolt and paper (the true
    edge). Shadows further away, or lighter than half-way, stay out."""
    if not core_mask.any():
        return core_mask
    bolt_brightness = float(np.median(gray_image[core_mask > 0]))
    halfway = (paper_background.astype(np.float32) + bolt_brightness) / 2
    grow_pixels = max(int(EDGE_GROW_MM * mat_layout.PIXELS_PER_MM), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_pixels + 1, 2 * grow_pixels + 1))
    near_core = cv2.dilate(core_mask, kernel) > 0
    rim_pixels = near_core & (gray_image.astype(np.float32) < halfway)
    return np.where((core_mask > 0) | rim_pixels, 255, 0).astype(np.uint8)


def build_object_mask(work_area_bgr):
    """Binary mask (255 = fastener) for the work area."""
    work_area_gray = cv2.cvtColor(work_area_bgr, cv2.COLOR_BGR2GRAY)
    work_area_gray = cv2.GaussianBlur(work_area_gray, (5, 5), 0)
    paper_background = estimate_paper_background(work_area_gray)
    darkness = cv2.subtract(paper_background, work_area_gray)
    required_darkness = np.maximum(DARKER_THAN_PAPER_BY, SHADOW_REJECT_FRACTION * paper_background.astype(np.float32))
    dark_pixels = darkness > required_darkness

    saturation = cv2.cvtColor(work_area_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    coloured_pixels = saturation > MINIMUM_RUST_SATURATION

    core_mask = np.where(dark_pixels | coloured_pixels, 255, 0).astype(np.uint8)
    object_mask = grow_core_to_true_edge(core_mask, work_area_gray, paper_background)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, small_kernel)
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, closing_kernel)
    return object_mask


MAXIMUM_BRIDGE_GAP_MM = 30.0


def long_axis_of_piece(contour):
    """(center, unit direction of the long side, short-side width) from the min-area rectangle."""
    (center_x, center_y), (side_a, side_b), angle_degrees = cv2.minAreaRect(contour)
    angle = np.deg2rad(angle_degrees)
    if side_a >= side_b:
        direction = np.array([np.cos(angle), np.sin(angle)])
        width = side_b
    else:
        direction = np.array([-np.sin(angle), np.cos(angle)])
        width = side_a
    return np.array([center_x, center_y]), direction, width


def piece_lies_on_axis(contour, center, direction, width):
    """True if another piece sits on the extension of the long piece's axis, within the gap limit.
    Returns (on_axis, near_end_point_of_other_piece_along_axis)."""
    points = contour.reshape(-1, 2).astype(np.float64) - center
    along = points @ direction
    across = points @ np.array([-direction[1], direction[0]])
    other_center_across = np.median(across)
    if abs(other_center_across) > width * 0.75:
        return False, None
    return True, (float(along.min()), float(along.max()))


def bridge_aligned_pieces(contours, image_shape):
    """A bright, shiny (zinc-plated) shank can be lighter than the detection threshold, splitting
    one bolt into a dark head and dark threads. Join pieces lying on the long piece's axis with a
    bar as wide as that piece. Pieces off the axis (a genuine second bolt) are left alone."""
    if len(contours) < 2:
        return contours
    long_piece = max(contours, key=lambda contour: max(cv2.minAreaRect(contour)[1]) / max(min(cv2.minAreaRect(contour)[1]), 1))
    center, direction, width = long_axis_of_piece(long_piece)
    long_points_along = (long_piece.reshape(-1, 2) - center) @ direction
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [long_piece], -1, 255, cv2.FILLED)
    remaining = []
    for contour in contours:
        if contour is long_piece:
            continue
        on_axis, other_span = piece_lies_on_axis(contour, center, direction, width)
        gap_limit = MAXIMUM_BRIDGE_GAP_MM * mat_layout.PIXELS_PER_MM
        if on_axis and other_span[0] - long_points_along.max() < gap_limit and long_points_along.min() - other_span[1] < gap_limit:
            draw_bridge(mask, center, direction, width, long_points_along, other_span)
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        else:
            remaining.append(contour)
    merged, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sorted(list(merged) + remaining, key=cv2.contourArea, reverse=True)


def draw_bridge(mask, center, direction, width, long_points_along, other_span):
    """Filled bar along the axis between the long piece's end and the other piece."""
    if other_span[0] > long_points_along.max():
        start, end = long_points_along.max() - 2, other_span[0] + 2
    else:
        start, end = other_span[1] - 2, long_points_along.min() + 2
    across = np.array([-direction[1], direction[0]]) * (width * 0.92 / 2)
    corners = [center + direction * start + across, center + direction * end + across,
               center + direction * end - across, center + direction * start - across]
    cv2.fillPoly(mask, [np.round(np.array(corners)).astype(np.int32)], 255)


def find_fastener_contours(rectified_bgr):
    """Return (contours_in_rectified_coordinates sorted largest first,
               touches_border flag for the largest one)."""
    x_min, y_min, x_max, y_max = mat_layout.WORK_AREA_MM
    inset = BORDER_INSET_MM
    scale = mat_layout.PIXELS_PER_MM
    offset_x = int((x_min + inset) * scale)
    offset_y = int((y_min + inset) * scale)
    work_area = rectified_bgr[offset_y:int((y_max - inset) * scale), offset_x:int((x_max - inset) * scale)]

    object_mask = build_object_mask(work_area)
    contours, _hierarchy = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    minimum_area_pixels = MINIMUM_OBJECT_AREA_MM2 * scale * scale
    kept_contours = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area_pixels]
    kept_contours.sort(key=cv2.contourArea, reverse=True)
    kept_contours = bridge_aligned_pieces(kept_contours, object_mask.shape)

    touches_border = False
    if kept_contours:
        box_x, box_y, box_width, box_height = cv2.boundingRect(kept_contours[0])
        touches_border = (box_x <= 1 or box_y <= 1 or
                          box_x + box_width >= work_area.shape[1] - 1 or
                          box_y + box_height >= work_area.shape[0] - 1)

    shifted_contours = [contour + np.array([[offset_x, offset_y]]) for contour in kept_contours]
    return shifted_contours, touches_border
