"""
Creates the printable RustReader measuring mat as a PDF (and a PNG preview).

Usage:
    python tools/make_marker_sheet.py

Print the PDF at 100% / "Actual size" (NOT "Fit to page"), then check the
100 mm ruler on the sheet with a real ruler.
"""
import os
import sys

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision import mat_layout  # noqa: E402

OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "..", "docs")


def get_marker_cell_pattern(marker_id):
    """Return a 6x6 grid of 0/1 values (1 = black) for one ArUco marker.
    4x4 data bits plus a 1-cell black border on every side."""
    dictionary = cv2.aruco.getPredefinedDictionary(mat_layout.ARUCO_DICTIONARY_ID)
    cells_per_side = 6
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, cells_per_side)
    pattern = []
    for row in range(cells_per_side):
        pattern_row = []
        for column in range(cells_per_side):
            is_black = marker_image[row, column] < 128
            pattern_row.append(1 if is_black else 0)
        pattern.append(pattern_row)
    return pattern


def draw_marker(pdf, marker_id):
    """Draw one marker as crisp vector squares at its exact mat position."""
    pattern = get_marker_cell_pattern(marker_id)
    cell_size_mm = mat_layout.MARKER_SIDE_MM / 6.0
    left_mm, top_mm = mat_layout.MARKER_TOP_LEFT_MM[marker_id]
    pdf.setFillColorRGB(0, 0, 0)
    for row in range(6):
        for column in range(6):
            if pattern[row][column] == 0:
                continue
            x_mm = left_mm + column * cell_size_mm
            # PDF y-axis starts at the bottom of the page, so flip it.
            y_mm = mat_layout.PAGE_HEIGHT_MM - (top_mm + (row + 1) * cell_size_mm)
            pdf.rect(x_mm * mm, y_mm * mm, cell_size_mm * mm, cell_size_mm * mm, stroke=0, fill=1)


def draw_work_area_and_text(pdf):
    """Draw the dashed work-area outline, the 100 mm check ruler and instructions."""
    x_min, y_min, x_max, y_max = mat_layout.WORK_AREA_MM
    page_height = mat_layout.PAGE_HEIGHT_MM
    pdf.setStrokeColorRGB(0.8, 0.8, 0.8)
    pdf.setDash(3, 3)
    pdf.rect(x_min * mm, (page_height - y_max) * mm, (x_max - x_min) * mm, (y_max - y_min) * mm, stroke=1, fill=0)
    pdf.setDash()
    pdf.setFillColorRGB(0.35, 0.35, 0.35)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawCentredString(105 * mm, (page_height - 26) * mm, "RustReader measuring mat")
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(105 * mm, (page_height - 32) * mm, "Place ONE bolt flat inside the dashed area.")
    pdf.drawCentredString(105 * mm, (page_height - 37) * mm, "Keep all 4 corner squares in the photo.")
    draw_check_ruler(pdf)


def draw_check_ruler(pdf):
    """A 100 mm line between the bottom markers to confirm the print scale."""
    page_height = mat_layout.PAGE_HEIGHT_MM
    ruler_y_mm = page_height - 262
    pdf.setStrokeColorRGB(0, 0, 0)
    pdf.setLineWidth(0.6)
    pdf.line(55 * mm, ruler_y_mm * mm, 155 * mm, ruler_y_mm * mm)
    for tick_mm in range(0, 101, 10):
        tick_height = 3 if tick_mm % 50 == 0 else 1.5
        pdf.line((55 + tick_mm) * mm, ruler_y_mm * mm, (55 + tick_mm) * mm, (ruler_y_mm + tick_height) * mm)
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(105 * mm, (ruler_y_mm - 5) * mm, "This line must measure exactly 100 mm. Print at 100% / Actual size.")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_FOLDER, "RustReader_Measuring_Mat_A4.pdf")
    pdf = canvas.Canvas(pdf_path, pagesize=A4)
    pdf.setTitle("RustReader measuring mat")
    for marker_id in mat_layout.MARKER_TOP_LEFT_MM:
        draw_marker(pdf, marker_id)
    draw_work_area_and_text(pdf)
    pdf.showPage()
    pdf.save()
    print("Saved", pdf_path)


if __name__ == "__main__":
    main()
