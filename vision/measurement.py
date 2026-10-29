"""
Measures one bolt from its outline in the straightened mat image.

Idea: rotate the bolt so it lies horizontally with the head on the LEFT,
then walk along it column by column recording how thick it is. That
"width profile" tells us where the head ends, the shank diameter and the
length. The same rotated image is reused for thread pitch.
"""
import math

import cv2
import numpy as np

from vision import mat_layout

HEAD_TO_SHANK_MINIMUM_RATIO = 1.15   # head must be at least 15% wider than shank
NUT_OR_WASHER_MAX_ASPECT = 1.6       # shorter than this (length / width) = not a bolt
TIP_TRIM_MM = 1.0                    # ignore the chamfered tip


def find_long_axis_angle_degrees(contour):
    """Angle of the bolt's long axis using PCA of its outline points."""
    points = contour.reshape(-1, 2).astype(np.float64)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    axis_x, axis_y = eigenvectors[0]
    return math.degrees(math.atan2(axis_y, axis_x))


def rotate_object_to_horizontal(rectified_bgr, contour):
    """Return (aligned_bgr, aligned_mask) with the bolt lying left-to-right."""
    box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
    padding = int(math.hypot(box_width, box_height) / 2) + 10
    height, width = rectified_bgr.shape[:2]
    left, top = max(box_x - padding, 0), max(box_y - padding, 0)
    right, bottom = min(box_x + box_width + padding, width), min(box_y + box_height + padding, height)

    region_bgr = rectified_bgr[top:bottom, left:right]
    region_mask = np.zeros(region_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(region_mask, [contour - np.array([[left, top]])], -1, 255, thickness=cv2.FILLED)

    angle_degrees = find_long_axis_angle_degrees(contour)
    center = (region_bgr.shape[1] / 2, region_bgr.shape[0] / 2)
    rotation = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    size = (region_bgr.shape[1], region_bgr.shape[0])
    aligned_bgr = cv2.warpAffine(region_bgr, rotation, size, flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    aligned_mask = cv2.warpAffine(region_mask, rotation, size, flags=cv2.INTER_NEAREST)
    return crop_to_mask(aligned_bgr, aligned_mask)


def crop_to_mask(aligned_bgr, aligned_mask):
    """Trim empty space around the object, keeping a few pixels of margin."""
    rows_with_object = np.where(aligned_mask.max(axis=1) > 0)[0]
    columns_with_object = np.where(aligned_mask.max(axis=0) > 0)[0]
    margin = 6
    top = max(rows_with_object[0] - margin, 0)
    bottom = min(rows_with_object[-1] + margin + 1, aligned_mask.shape[0])
    left = columns_with_object[0]
    right = columns_with_object[-1] + 1
    return aligned_bgr[top:bottom, left:right], aligned_mask[top:bottom, left:right]


def compute_width_profile(aligned_mask):
    """For every column: (top_row, bottom_row, width_in_pixels)."""
    column_count = aligned_mask.shape[1]
    top_rows = np.zeros(column_count)
    bottom_rows = np.zeros(column_count)
    widths = np.zeros(column_count)
    for column in range(column_count):
        object_rows = np.nonzero(aligned_mask[:, column])[0]
        if len(object_rows) == 0:
            continue
        top_rows[column] = object_rows[0]
        bottom_rows[column] = object_rows[-1]
        widths[column] = object_rows[-1] - object_rows[0] + 1
    return top_rows, bottom_rows, widths


def put_head_on_left(aligned_bgr, aligned_mask):
    """Flip left-right if the wider end (the head) is on the right."""
    _top, _bottom, widths = compute_width_profile(aligned_mask)
    end_length = max(int(len(widths) * 0.15), 1)
    left_end_width = widths[:end_length].mean()
    right_end_width = widths[-end_length:].mean()
    if right_end_width > left_end_width:
        return cv2.flip(aligned_bgr, 1), cv2.flip(aligned_mask, 1)
    return aligned_bgr, aligned_mask


def find_head_end_column(widths):
    """Column where the head stops. Returns (head_end_column, head_found)."""
    column_count = len(widths)
    shank_typical_width = np.median(widths[int(column_count * 0.4):int(column_count * 0.9)])
    head_typical_width = np.percentile(widths[:max(int(column_count * 0.1), 1)], 75)
    if head_typical_width < shank_typical_width * HEAD_TO_SHANK_MINIMUM_RATIO:
        return 0, False
    halfway_width = (head_typical_width + shank_typical_width) / 2
    for column in range(column_count):
        if widths[column] < halfway_width and column > column_count * 0.02:
            return column, True
    return 0, False


def looks_like_nut_or_washer(widths):
    """Short, and no narrow shank: a nut or washer lying on the mat."""
    total_length = len(widths)
    maximum_width = widths.max()
    typical_to_maximum = np.median(widths[widths > 0]) / maximum_width
    return total_length / maximum_width < NUT_OR_WASHER_MAX_ASPECT and typical_to_maximum > 0.8


def find_shank_start_column(widths, head_end_column):
    """First column after the head where the outline is down to shank width.
    Matters for countersunk heads, whose cone runs well past the halfway point."""
    column_count = len(widths)
    shank_typical_width = np.median(widths[int(column_count * 0.5):int(column_count * 0.9)])
    for column in range(head_end_column, column_count):
        if widths[column] <= shank_typical_width * 1.1:
            return column
    return head_end_column


def classify_head_shape(widths, shank_start_column, diameter_pixels):
    """hex / socket (tall flat head) / countersunk (cone) / flanged (wider at the bottom)."""
    top_part = widths[:max(int(shank_start_column * 0.3), 2)]
    lower_part = widths[int(shank_start_column * 0.6):max(int(shank_start_column * 0.85), int(shank_start_column * 0.6) + 1)]
    top_width = np.percentile(top_part, 75)
    lower_width = np.median(lower_part)
    if lower_width < top_width * 0.85:
        return "countersunk"
    if lower_width > top_width * 1.12:
        return "flanged"
    if shank_start_column / diameter_pixels > 0.85:
        return "socket"
    return "hex"


def measure_bolt(rectified_bgr, contour):
    """Measure one fastener. Returns (measurements_dict, aligned_bgr, aligned_mask)."""
    aligned_bgr, aligned_mask = rotate_object_to_horizontal(rectified_bgr, contour)
    aligned_bgr, aligned_mask = put_head_on_left(aligned_bgr, aligned_mask)
    _top, _bottom, widths = compute_width_profile(aligned_mask)
    pixels_per_mm = mat_layout.PIXELS_PER_MM

    total_length_mm = len(widths) / pixels_per_mm
    if looks_like_nut_or_washer(widths):
        measurements = {"object_type": "nut_or_washer", "total_length_mm": round(total_length_mm, 2),
                        "maximum_width_mm": round(widths.max() / pixels_per_mm, 2)}
        return measurements, aligned_bgr, aligned_mask

    head_end_column, head_found = find_head_end_column(widths)
    measurements = summarise_shank(widths, head_end_column, head_found)
    measurements["total_length_mm"] = round(total_length_mm, 2)
    return measurements, aligned_bgr, aligned_mask


def summarise_shank(widths, head_end_column, head_found):
    """Diameter, length and head size from the width profile."""
    pixels_per_mm = mat_layout.PIXELS_PER_MM
    shank_start = find_shank_start_column(widths, head_end_column) if head_found else 0
    shank_start += int(0.5 * pixels_per_mm)
    shank_end = len(widths) - int(TIP_TRIM_MM * pixels_per_mm)
    shank_widths = widths[shank_start:shank_end]
    diameter_pixels = float(np.percentile(shank_widths, 90))

    measurements = {
        "object_type": "bolt" if head_found else "stud_or_rod",
        "head_type": "none",
        "shank_start_column": int(shank_start),
        "shank_end_column": int(shank_end),
        "major_diameter_mm": round(diameter_pixels / pixels_per_mm, 2),
        "minor_diameter_mm": round(float(np.percentile(shank_widths, 10)) / pixels_per_mm, 2),
        "length_under_head_mm": round((len(widths) - head_end_column) / pixels_per_mm, 2),
    }
    if head_found:
        add_head_measurements(measurements, widths, head_end_column, shank_start, diameter_pixels)
    return measurements


def add_head_measurements(measurements, widths, head_end_column, shank_start, diameter_pixels):
    """Head type, size and the correct nominal length (countersunk length includes the head)."""
    pixels_per_mm = mat_layout.PIXELS_PER_MM
    head_type = classify_head_shape(widths, shank_start, diameter_pixels)
    measurements["head_type"] = head_type
    measurements["head_side_width_mm"] = round(float(np.percentile(widths[:head_end_column], 75)) / pixels_per_mm, 2)
    measurements["head_height_mm"] = round(head_end_column / pixels_per_mm, 2)
    if head_type == "countersunk":
        measurements["length_under_head_mm"] = round(len(widths) / pixels_per_mm, 2)
