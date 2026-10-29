"""
Estimates thread pitch (mm between thread peaks) from the bolt's silhouette.

Seen from the side, a threaded shank has a saw-tooth edge. We record the
edge position along the shank, remove the slow trend, and use an FFT to find
the strongest repeating distance. That distance is the pitch.

Two ways of finding the edge (the agent picks which to use):
  "mask"     - edge of the binary outline. Fast; fine for clean threads.
  "enhanced" - CLAHE contrast boost + vertical gradient with sub-pixel peak.
               Slower, but recovers thread crests partly hidden by rust.
"""
import cv2
import numpy as np

from vision import mat_layout

SMALLEST_PITCH_MM = 0.4
LARGEST_PITCH_MM = 3.5
TREND_WINDOW_MM = 4.0
EDGE_SEARCH_MM = 1.5


def edge_rows_from_mask(aligned_mask, start_column, end_column):
    """Top and bottom edge row of the outline for each column."""
    top_edges = []
    bottom_edges = []
    for column in range(start_column, end_column):
        object_rows = np.nonzero(aligned_mask[:, column])[0]
        if len(object_rows) == 0:
            top_edges.append(np.nan)
            bottom_edges.append(np.nan)
            continue
        top_edges.append(object_rows[0])
        bottom_edges.append(object_rows[-1])
    return np.array(top_edges, dtype=float), np.array(bottom_edges, dtype=float)


def strongest_edge_near(gradient_column, expected_row, search_pixels):
    """Sub-pixel row of the strongest gradient within +/- search_pixels."""
    first_row = max(int(expected_row - search_pixels), 1)
    last_row = min(int(expected_row + search_pixels), len(gradient_column) - 2)
    if last_row <= first_row:
        return float(expected_row)
    window = gradient_column[first_row:last_row + 1]
    peak_row = first_row + int(np.argmax(window))
    left_value = gradient_column[peak_row - 1]
    center_value = gradient_column[peak_row]
    right_value = gradient_column[peak_row + 1]
    curvature = left_value - 2 * center_value + right_value
    if curvature == 0:
        return float(peak_row)
    return peak_row + 0.5 * (left_value - right_value) / curvature


def edge_rows_from_enhanced_image(aligned_bgr, aligned_mask, start_column, end_column):
    """Like edge_rows_from_mask, but snaps each edge to the strongest
    brightness change in a contrast-enhanced copy of the image."""
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    contrast_enhancer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = contrast_enhancer.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 5, 40, 5)
    vertical_gradient = np.abs(cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3))

    mask_top, mask_bottom = edge_rows_from_mask(aligned_mask, start_column, end_column)
    search_pixels = EDGE_SEARCH_MM * mat_layout.PIXELS_PER_MM
    top_edges = []
    bottom_edges = []
    for index, column in enumerate(range(start_column, end_column)):
        if np.isnan(mask_top[index]):
            top_edges.append(np.nan)
            bottom_edges.append(np.nan)
            continue
        top_edges.append(strongest_edge_near(vertical_gradient[:, column], mask_top[index], search_pixels))
        bottom_edges.append(strongest_edge_near(vertical_gradient[:, column], mask_bottom[index], search_pixels))
    return np.array(top_edges), np.array(bottom_edges)


def remove_slow_trend(edge_rows, trend_window_mm=TREND_WINDOW_MM):
    """Subtract a moving average so only the thread ripple remains."""
    filled = edge_rows.copy()
    missing = np.isnan(filled)
    if missing.all():
        return np.zeros_like(filled)
    filled[missing] = np.nanmean(filled)
    window = max(int(trend_window_mm * mat_layout.PIXELS_PER_MM) | 1, 3)
    trend = np.convolve(filled, np.ones(window) / window, mode="same")
    ripple = filled - trend
    half = window // 2
    return ripple[half:len(ripple) - half]   # edges of the moving average are unreliable


def find_threaded_section(ripple):
    """Keep the tip-side stretch where the ripple is strong (the threads).
    Long bolts are often only threaded near the tip."""
    window = max(int(3 * mat_layout.PIXELS_PER_MM), 5)
    if len(ripple) <= window * 2:
        return ripple
    local_strength = np.sqrt(np.convolve(ripple ** 2, np.ones(window) / window, mode="same"))
    strong = local_strength > 0.5 * np.percentile(local_strength, 95)
    first_threaded = len(ripple) - 1
    while first_threaded > 0 and strong[first_threaded - 1]:
        first_threaded -= 1
    if len(ripple) - first_threaded < window * 2:
        return ripple
    return ripple[first_threaded:]


def find_dominant_period(ripple, largest_pitch_mm=LARGEST_PITCH_MM):
    """Return (period_in_pixels, confidence) using an FFT of the ripple."""
    pixels_per_mm = mat_layout.PIXELS_PER_MM
    windowed = ripple * np.hanning(len(ripple))
    spectrum_length = 16384
    magnitudes = np.abs(np.fft.rfft(windowed, n=spectrum_length))
    frequencies = np.fft.rfftfreq(spectrum_length)             # cycles per pixel

    lowest_frequency = 1.0 / (largest_pitch_mm * pixels_per_mm)
    highest_frequency = 1.0 / (SMALLEST_PITCH_MM * pixels_per_mm)
    in_band = (frequencies >= lowest_frequency) & (frequencies <= highest_frequency)
    band_indices = np.nonzero(in_band)[0]
    peak_index = band_indices[np.argmax(magnitudes[band_indices])]
    confidence = magnitudes[peak_index] / (np.median(magnitudes[band_indices]) + 1e-9)
    return float(1.0 / frequencies[peak_index]), float(confidence)


def largest_plausible_pitch_mm(diameter_mm):
    """Coarsest standard pitch is about 0.15 x diameter (M6: 1.0, M24: 3.0). Allowing a
    margin above that lets short screws keep a usable FFT window."""
    if not diameter_mm:
        return LARGEST_PITCH_MM
    return min(LARGEST_PITCH_MM, 0.2 * diameter_mm + 0.6)


def estimate_thread_pitch(aligned_bgr, aligned_mask, shank_start_column, shank_end_column, edge_mode="mask",
                          diameter_mm=None):
    """Return dict with pitch_mm, confidence, threaded_length_mm and edge_mode used."""
    largest_pitch_mm = largest_plausible_pitch_mm(diameter_mm)
    trend_window_mm = min(TREND_WINDOW_MM, max(2.5, 2.2 * largest_pitch_mm))
    if edge_mode == "enhanced":
        top_edges, bottom_edges = edge_rows_from_enhanced_image(aligned_bgr, aligned_mask,
                                                                shank_start_column, shank_end_column)
    else:
        top_edges, bottom_edges = edge_rows_from_mask(aligned_mask, shank_start_column, shank_end_column)

    top_ripple = find_threaded_section(remove_slow_trend(top_edges, trend_window_mm))
    bottom_ripple = find_threaded_section(remove_slow_trend(bottom_edges, trend_window_mm))
    minimum_samples = int(3 * largest_pitch_mm * mat_layout.PIXELS_PER_MM)
    if len(top_ripple) < minimum_samples or len(bottom_ripple) < minimum_samples:
        return {"pitch_mm": None, "confidence": 0.0, "threaded_length_mm": 0.0, "edge_mode": edge_mode}

    top_period, top_confidence = find_dominant_period(top_ripple, largest_pitch_mm)
    bottom_period, bottom_confidence = find_dominant_period(bottom_ripple, largest_pitch_mm)
    if top_confidence >= bottom_confidence:
        best_period, best_confidence, threaded_pixels = top_period, top_confidence, len(top_ripple)
    else:
        best_period, best_confidence, threaded_pixels = bottom_period, bottom_confidence, len(bottom_ripple)
    return {"pitch_mm": round(best_period / mat_layout.PIXELS_PER_MM, 3),
            "confidence": round(best_confidence, 2),
            "threaded_length_mm": round(threaded_pixels / mat_layout.PIXELS_PER_MM, 1),
            "edge_mode": edge_mode}
