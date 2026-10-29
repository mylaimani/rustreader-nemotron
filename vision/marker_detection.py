"""
Step 1 of the pipeline: find the 4 mat markers in the phone photo and use
them to build a straight, top-down image where 1 mm = PIXELS_PER_MM pixels.
"""
import cv2
import numpy as np

from vision import mat_layout


def detect_mat_markers(photo_bgr):
    """Return {marker_id: 4x2 array of corner pixels} for every mat marker found."""
    photo_gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(mat_layout.ARUCO_DICTIONARY_ID)
    detector_parameters = cv2.aruco.DetectorParameters()
    detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters)
    corner_sets, marker_ids, _rejected = detector.detectMarkers(photo_gray)

    found_markers = {}
    if marker_ids is None:
        return found_markers
    for corners, marker_id in zip(corner_sets, marker_ids.ravel()):
        marker_id = int(marker_id)
        if marker_id in mat_layout.MARKER_TOP_LEFT_MM:
            found_markers[marker_id] = corners.reshape(4, 2)
    return found_markers


def estimate_photo_pixels_per_mm(found_markers):
    """How many photo pixels cover 1 mm (average marker side length / 30 mm).
    Tells the agent whether the phone was close enough to see threads."""
    side_lengths = []
    for corners in found_markers.values():
        for corner_index in range(4):
            next_index = (corner_index + 1) % 4
            side_lengths.append(np.linalg.norm(corners[corner_index] - corners[next_index]))
    if not side_lengths:
        return 0.0
    return float(np.mean(side_lengths) / mat_layout.MARKER_SIDE_MM)


def compute_photo_to_mat_homography(found_markers):
    """Return (homography, reprojection_error_mm). Needs at least 2 markers."""
    photo_points = []
    mat_points = []
    for marker_id, corners in found_markers.items():
        paper_corners_mm = mat_layout.marker_corners_on_paper_mm(marker_id)
        for corner_index in range(4):
            photo_points.append(corners[corner_index])
            x_mm, y_mm = paper_corners_mm[corner_index]
            mat_points.append([x_mm * mat_layout.PIXELS_PER_MM, y_mm * mat_layout.PIXELS_PER_MM])
    photo_points = np.array(photo_points, dtype=np.float32)
    mat_points = np.array(mat_points, dtype=np.float32)

    homography, _inlier_flags = cv2.findHomography(photo_points, mat_points, cv2.RANSAC, 3.0)
    if homography is None:
        return None, None
    projected_points = cv2.perspectiveTransform(photo_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error_pixels = np.linalg.norm(projected_points - mat_points, axis=1).mean()
    reprojection_error_mm = float(error_pixels / mat_layout.PIXELS_PER_MM)
    return homography, reprojection_error_mm


def rectify_photo_to_mat(photo_bgr, homography):
    """Warp the photo into a flat top-down view of the whole A4 mat."""
    output_width = int(mat_layout.PAGE_WIDTH_MM * mat_layout.PIXELS_PER_MM)
    output_height = int(mat_layout.PAGE_HEIGHT_MM * mat_layout.PIXELS_PER_MM)
    return cv2.warpPerspective(photo_bgr, homography, (output_width, output_height),
                               flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
