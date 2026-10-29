"""Draws measurement lines on the aligned bolt image, for the web page and report."""
import cv2

from vision import mat_layout


def draw_measurement_overlay(aligned_bgr, measurements):
    """Return a copy with the head boundary and shank diameter marked."""
    annotated = aligned_bgr.copy()
    if measurements.get("object_type") != "bolt":
        return annotated
    height = annotated.shape[0]
    pixels_per_mm = mat_layout.PIXELS_PER_MM
    head_end_column = int(measurements["head_height_mm"] * pixels_per_mm)
    cv2.line(annotated, (head_end_column, 0), (head_end_column, height - 1), (0, 200, 255), 2)

    center_row = height // 2
    half_diameter = int(measurements["major_diameter_mm"] * pixels_per_mm / 2)
    shank_start = measurements["shank_start_column"]
    shank_end = measurements["shank_end_column"]
    for row in (center_row - half_diameter, center_row + half_diameter):
        cv2.line(annotated, (shank_start, row), (shank_end, row), (80, 220, 80), 1)
    return annotated
