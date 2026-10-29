"""
Rust severity and the "Rust Mode" enhanced view.

Rust pixels = orange-to-brown hue with real colour saturation, inside the
bolt outline. The fraction of rusty pixels gives a severity level.
This is ADVISORY only: a photo cannot see fatigue, stretch or thread damage.
"""
import cv2
import numpy as np

RUST_HUE_LOW = 0          # OpenCV hue runs 0-180
RUST_HUE_HIGH = 25
RUST_RED_HUE_LOW = 170
MINIMUM_SATURATION = 70
MINIMUM_VALUE = 35
MAXIMUM_VALUE = 235


def measure_rust(aligned_bgr, aligned_mask):
    """Return dict with rust_fraction (0-1) and rust_level."""
    hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    rust_coloured = ((hue <= RUST_HUE_HIGH) | (hue >= RUST_RED_HUE_LOW)) & (hue >= RUST_HUE_LOW)
    rust_coloured &= (saturation >= MINIMUM_SATURATION)
    rust_coloured &= (value >= MINIMUM_VALUE) & (value <= MAXIMUM_VALUE)
    inside_bolt = aligned_mask > 0

    bolt_pixel_count = int(inside_bolt.sum())
    if bolt_pixel_count == 0:
        return {"rust_fraction": 0.0, "rust_level": "unknown"}
    rust_fraction = float((rust_coloured & inside_bolt).sum()) / bolt_pixel_count
    if rust_fraction < 0.05:
        rust_level = "none"
    elif rust_fraction < 0.20:
        rust_level = "light"
    elif rust_fraction < 0.45:
        rust_level = "moderate"
    else:
        rust_level = "heavy"
    return {"rust_fraction": round(rust_fraction, 3), "rust_level": rust_level}


def build_rust_mode_view(aligned_bgr):
    """Contrast-boosted, sharpened copy that makes worn threads/markings easier to see."""
    lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    contrast_enhancer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lightness = contrast_enhancer.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge([lightness, channel_a, channel_b]), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    return cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)


def build_rust_overlay(aligned_bgr, aligned_mask):
    """Copy of the image with rusty pixels tinted orange, for the report/demo."""
    hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    rust_pixels = ((hue <= RUST_HUE_HIGH) | (hue >= RUST_RED_HUE_LOW)) & (saturation >= MINIMUM_SATURATION)
    rust_pixels &= (value >= MINIMUM_VALUE) & (value <= MAXIMUM_VALUE) & (aligned_mask > 0)
    overlay = aligned_bgr.copy()
    overlay[rust_pixels] = (0.4 * overlay[rust_pixels] + 0.6 * np.array([0, 81, 230])).astype(np.uint8)
    return overlay
