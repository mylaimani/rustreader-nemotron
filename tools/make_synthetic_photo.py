"""
Makes a fake "phone photo" of a bolt on the measuring mat, with known size,
pitch and rust. Used to test the pipeline when no real photos are at hand,
and to produce controlled failure cases (blur, missing corner, two bolts).

Usage:
    python tools/make_synthetic_photo.py --diameter 8 --pitch 1.25 --length 40 \
        --head 13 --rust 0.6 --angle 25 --output samples/synthetic_M8x40.jpg
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision import mat_layout  # noqa: E402

RENDER_PIXELS_PER_MM = 16.0
HEAD_TYPE = "hex"   # set from --head-type


def render_blank_mat():
    """White A4 page with the 4 markers, at RENDER_PIXELS_PER_MM."""
    width = int(mat_layout.PAGE_WIDTH_MM * RENDER_PIXELS_PER_MM)
    height = int(mat_layout.PAGE_HEIGHT_MM * RENDER_PIXELS_PER_MM)
    page = np.full((height, width, 3), 245, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(mat_layout.ARUCO_DICTIONARY_ID)
    marker_pixels = int(mat_layout.MARKER_SIDE_MM * RENDER_PIXELS_PER_MM)
    for marker_id, (left_mm, top_mm) in mat_layout.MARKER_TOP_LEFT_MM.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_pixels)
        left = int(left_mm * RENDER_PIXELS_PER_MM)
        top = int(top_mm * RENDER_PIXELS_PER_MM)
        page[top:top + marker_pixels, left:left + marker_pixels] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return page


def bolt_outline_mm(diameter, pitch, length, head_width, head_height, threaded_length):
    """Side-view outline of a hex bolt, head at x=0, as a list of (x, y) mm points."""
    thread_depth = 0.6 * pitch
    shank_start = head_height
    shank_end = head_height + length
    thread_start = shank_end - threaded_length
    top_edge = []
    bottom_edge = []
    x = shank_start
    while x <= shank_end:
        if x < thread_start:
            top_offset, bottom_offset = 0.0, 0.0
        else:
            phase = ((x - thread_start) / pitch) % 1.0
            top_offset = thread_depth * (1 - abs(2 * phase - 1))
            bottom_offset = thread_depth * (1 - abs(2 * ((phase + 0.5) % 1.0) - 1))
        top_edge.append((x, -diameter / 2 + top_offset))
        bottom_edge.append((x, diameter / 2 - bottom_offset))
        x += 0.04
    if HEAD_TYPE == "countersunk":
        # 90 degree cone from head_width down to the shank; length INCLUDES the head
        cone_height = (head_width - diameter) / 2
        shift = cone_height - head_height
        top_edge = [(x + shift, y) for x, y in top_edge if x + shift >= cone_height]
        bottom_edge = [(x + shift, y) for x, y in bottom_edge if x + shift >= cone_height]
        head = [(0, head_width / 2), (0, -head_width / 2)]
        return head + top_edge + list(reversed(bottom_edge))
    head = [(0, head_width / 2), (0, -head_width / 2), (head_height, -head_width / 2)]
    return head + top_edge + list(reversed(bottom_edge)) + [(head_height, head_width / 2)]


def draw_bolt(page, outline_mm, center_mm, angle_degrees, rust_amount):
    """Rotate/translate the outline onto the page and paint it steel/rust coloured."""
    angle = math.radians(angle_degrees)
    points = np.array(outline_mm)
    points[:, 0] -= points[:, 0].mean()
    rotated_x = points[:, 0] * math.cos(angle) - points[:, 1] * math.sin(angle)
    rotated_y = points[:, 0] * math.sin(angle) + points[:, 1] * math.cos(angle)
    pixel_points = np.stack([(rotated_x + center_mm[0]) * RENDER_PIXELS_PER_MM,
                             (rotated_y + center_mm[1]) * RENDER_PIXELS_PER_MM], axis=1)
    mask = np.zeros(page.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pixel_points).astype(np.int32)], 255, lineType=cv2.LINE_AA)

    random_generator = np.random.default_rng(7)
    steel = np.array([95, 95, 100], dtype=float)
    rust = np.array([25, 75, 150], dtype=float)                        # BGR orange-brown
    blotches = cv2.GaussianBlur(random_generator.random(page.shape[:2]).astype(np.float32), (0, 0), 6)
    blotches = (blotches - blotches.min()) / (blotches.max() - blotches.min() + 1e-9)
    rust_weight = np.clip((blotches - (1 - rust_amount)) * 4, 0, 1)[..., None]
    colour = steel * (1 - rust_weight) + rust * rust_weight
    colour += random_generator.normal(0, 8, page.shape)
    alpha = (mask.astype(float) / 255.0)[..., None]
    page[:] = (page * (1 - alpha) + np.clip(colour, 0, 255) * alpha).astype(np.uint8)


def photograph_page(page, tilt, blur_sigma, hide_marker_id):
    """Warp the flat page into a 3024x4032 'phone photo' with perspective."""
    if hide_marker_id is not None:
        left_mm, top_mm = mat_layout.MARKER_TOP_LEFT_MM[hide_marker_id]
        left, top = int((left_mm - 3) * RENDER_PIXELS_PER_MM), int((top_mm - 3) * RENDER_PIXELS_PER_MM)
        size = int((mat_layout.MARKER_SIDE_MM + 6) * RENDER_PIXELS_PER_MM)
        page[top:top + size, left:left + size] = (60, 90, 120)       # e.g. a thumb or the table
    height, width = page.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = np.float32([[260 + tilt, 200], [2780 - tilt, 240], [2900, 3850], [130, 3800]])
    homography = cv2.getPerspectiveTransform(source, destination)
    photo = cv2.warpPerspective(page, homography, (3024, 4032), flags=cv2.INTER_AREA,
                                borderValue=(70, 105, 140))
    if blur_sigma > 0:
        photo = cv2.GaussianBlur(photo, (0, 0), blur_sigma)
    noise = np.random.default_rng(3).normal(0, 3, photo.shape)
    return np.clip(photo + noise, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diameter", type=float, default=8.0)
    parser.add_argument("--pitch", type=float, default=1.25)
    parser.add_argument("--length", type=float, default=40.0)
    parser.add_argument("--head", type=float, default=13.0, help="head width across flats, mm")
    parser.add_argument("--threaded", type=float, default=0, help="threaded length, 0 = fully threaded")
    parser.add_argument("--rust", type=float, default=0.5, help="0 = clean steel, 1 = fully rusted")
    parser.add_argument("--angle", type=float, default=20.0)
    parser.add_argument("--tilt", type=float, default=120.0)
    parser.add_argument("--blur", type=float, default=0.8)
    parser.add_argument("--hide-marker", type=int, default=None)
    parser.add_argument("--second-bolt", action="store_true")
    parser.add_argument("--head-type", default="hex", choices=["hex", "socket", "countersunk"])
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    page = render_blank_mat()
    global HEAD_TYPE
    HEAD_TYPE = arguments.head_type
    head_height = arguments.diameter * (1.0 if HEAD_TYPE == "socket" else 0.65)
    threaded_length = arguments.threaded or 1000.0
    shank_length = arguments.length
    if HEAD_TYPE == "countersunk":
        shank_length -= (arguments.head - arguments.diameter) / 2   # nominal length includes the cone
    outline = bolt_outline_mm(arguments.diameter, arguments.pitch, shank_length,
                              arguments.head, head_height, threaded_length)
    HEAD_TYPE = "hex"   # the optional second bolt is always hex
    draw_bolt(page, outline, (105, 140), arguments.angle, arguments.rust)
    if arguments.second_bolt:
        second_outline = bolt_outline_mm(6, 1.0, 25, 10, 4, 25)
        draw_bolt(page, second_outline, (105, 215), -10, 0.3)
    photo = photograph_page(page, arguments.tilt, arguments.blur, arguments.hide_marker)
    cv2.imwrite(arguments.output, photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("Saved", arguments.output)


if __name__ == "__main__":
    main()
