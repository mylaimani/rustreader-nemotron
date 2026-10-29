"""
Checks whether the photo is good enough to measure. The agent uses these
numbers to decide whether to ask the user for another photo.
"""
import cv2
import numpy as np

from vision import mat_layout

# Below this, marker edges are too soft (blurry photo). The value is the Laplacian variance
# divided by the marker's contrast squared (x1000), so grey-printed markers are not penalised.
# Measured: blurry photos 0.3-0.4, sharp real/synthetic photos 4.8-18.
MINIMUM_MARKER_SHARPNESS = 1.5
# Below this, the paper looks grey instead of white (photo too dark).
MINIMUM_PAPER_BRIGHTNESS = 90.0


def crop_mm_region(rectified_image, left_mm, top_mm, right_mm, bottom_mm):
    """Cut a rectangle (given in mm) out of the straightened mat image."""
    scale = mat_layout.PIXELS_PER_MM
    return rectified_image[int(top_mm * scale):int(bottom_mm * scale),
                           int(left_mm * scale):int(right_mm * scale)]


def measure_marker_sharpness(rectified_bgr, photo_pixels_per_mm=mat_layout.PIXELS_PER_MM):
    """Variance of the Laplacian over the marker squares. Markers are pure
    black/white, so soft edges here mean the whole photo is blurred.
    Measured at the photo's own resolution: a small photo is upscaled when
    straightened, and that softening must not count as blur."""
    shrink = min(1.0, photo_pixels_per_mm / mat_layout.PIXELS_PER_MM)
    sharpness_values = []
    for left_mm, top_mm in mat_layout.MARKER_TOP_LEFT_MM.values():
        right_mm = left_mm + mat_layout.MARKER_SIDE_MM
        bottom_mm = top_mm + mat_layout.MARKER_SIDE_MM
        marker_crop = crop_mm_region(rectified_bgr, left_mm, top_mm, right_mm, bottom_mm)
        marker_gray = cv2.cvtColor(marker_crop, cv2.COLOR_BGR2GRAY)
        if shrink < 1.0:
            marker_gray = cv2.resize(marker_gray, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        laplacian_variance = cv2.Laplacian(marker_gray, cv2.CV_64F).var()
        contrast = np.percentile(marker_gray, 95) - np.percentile(marker_gray, 5)
        sharpness_values.append(1000.0 * laplacian_variance / (contrast ** 2 + 1e-6))
    return float(np.median(sharpness_values))


def measure_paper_brightness(rectified_bgr):
    """Median brightness of the white work area (0-255)."""
    x_min, y_min, x_max, y_max = mat_layout.WORK_AREA_MM
    work_area = crop_mm_region(rectified_bgr, x_min, y_min, x_max, y_max)
    work_area_gray = cv2.cvtColor(work_area, cv2.COLOR_BGR2GRAY)
    return float(np.median(work_area_gray))


def assess_image_quality(rectified_bgr, photo_pixels_per_mm=mat_layout.PIXELS_PER_MM):
    """Return a dict with the quality numbers and a list of problems found."""
    sharpness = measure_marker_sharpness(rectified_bgr, photo_pixels_per_mm)
    brightness = measure_paper_brightness(rectified_bgr)
    problems = []
    if sharpness < MINIMUM_MARKER_SHARPNESS:
        problems.append("blurry")
    if brightness < MINIMUM_PAPER_BRIGHTNESS:
        problems.append("too_dark")
    return {"marker_sharpness": round(sharpness, 2),
            "paper_brightness": round(brightness, 1),
            "problems": problems}
