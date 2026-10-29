"""
Matches measured numbers against the fastener spec table.
Values are the same as rustreader_new/lib/data/fastener_spec_data.dart.

Each candidate gets an error score: how many "tolerances" away it is on
diameter, pitch and head size. Lowest score wins. If the top two are close,
the match is ambiguous and the agent asks a human to confirm.
"""
import math

# designation, thread system, diameter mm, pitch mm, wrench sizes (ISO first, DIN alternative)
FASTENER_SPECS = [
    ("M3", "metric coarse", 3.0, 0.5, [5.5]),
    ("M4", "metric coarse", 4.0, 0.7, [7.0]),
    ("M5", "metric coarse", 5.0, 0.8, [8.0]),
    ("M6", "metric coarse", 6.0, 1.0, [10.0]),
    ("M8", "metric coarse", 8.0, 1.25, [13.0]),
    ("M10", "metric coarse", 10.0, 1.5, [16.0, 17.0]),
    ("M12", "metric coarse", 12.0, 1.75, [18.0, 19.0]),
    ("M14", "metric coarse", 14.0, 2.0, [21.0, 22.0]),
    ("M16", "metric coarse", 16.0, 2.0, [24.0]),
    ("M18", "metric coarse", 18.0, 2.5, [27.0]),
    ("M20", "metric coarse", 20.0, 2.5, [30.0]),
    ("M24", "metric coarse", 24.0, 3.0, [36.0]),
    ("M8x1.0", "metric fine", 8.0, 1.0, [13.0]),
    ("M10x1.25", "metric fine", 10.0, 1.25, [16.0, 17.0]),
    ("M10x1.0", "metric fine", 10.0, 1.0, [16.0, 17.0]),
    ("M12x1.5", "metric fine", 12.0, 1.5, [18.0, 19.0]),
    ("M12x1.25", "metric fine", 12.0, 1.25, [18.0, 19.0]),
    ("M14x1.5", "metric fine", 14.0, 1.5, [21.0, 22.0]),
    ("M16x1.5", "metric fine", 16.0, 1.5, [24.0]),
    ("1/4-20 UNC", "UNC", 6.35, 1.27, [11.11]),
    ("5/16-18 UNC", "UNC", 7.94, 1.41, [12.7]),
    ("3/8-16 UNC", "UNC", 9.53, 1.59, [14.29]),
    ("7/16-14 UNC", "UNC", 11.11, 1.81, [15.88]),
    ("1/2-13 UNC", "UNC", 12.7, 1.95, [19.05]),
]

METRIC_STANDARD_LENGTHS_MM = [6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
                              90, 100, 110, 120, 130, 140, 150, 160, 180, 200]
INCH_STANDARD_LENGTHS_MM = [round(eighths * 25.4 / 8, 1) for eighths in range(4, 65, 2)]

DIAMETER_TOLERANCE_MM = 0.35
PITCH_TOLERANCE_MM = 0.08
HEAD_TOLERANCE_MM = 1.0
CORNERS_TO_FLATS_RATIO = 2 / math.sqrt(3)     # a hex seen from the side shows flats OR corners
AMBIGUOUS_IF_PROBABILITY_BELOW = 0.6

# Hex key (Allen) sizes: ISO 4762 socket-head cap screws and ISO 10642 countersunk screws
SOCKET_HEAD_KEY_MM = {"M3": 2.5, "M4": 3, "M5": 4, "M6": 5, "M8": 6, "M10": 8, "M12": 10, "M14": 12, "M16": 14, "M20": 17}
COUNTERSUNK_KEY_MM = {"M3": 2, "M4": 2.5, "M5": 3, "M6": 4, "M8": 5, "M10": 6, "M12": 8, "M16": 10, "M20": 12}
PART_NAMES = {"hex": "hex bolt", "socket": "socket-head cap screw", "countersunk": "countersunk screw",
              "flanged": "flange bolt", "none": "stud"}


def describe_tool(designation, head_type, wrench_size_mm):
    """Which tool fits: spanner for hex/flanged heads, Allen key for socket/countersunk."""
    base_size = designation.split("x")[0]
    if head_type == "socket" and base_size in SOCKET_HEAD_KEY_MM:
        return f"Allen key {SOCKET_HEAD_KEY_MM[base_size]} mm"
    if head_type == "countersunk" and base_size in COUNTERSUNK_KEY_MM:
        return f"Allen key {COUNTERSUNK_KEY_MM[base_size]} mm"
    if head_type in ("hex", "flanged"):
        return f"spanner {wrench_size_mm:g} mm"
    return "tool size not known"


def score_candidate(spec, measurements, pitch_result):
    """Lower is better. Diameter always counts; pitch and head only when we trust them."""
    _designation, _system, nominal_diameter, nominal_pitch, wrench_sizes = spec
    diameter_error = (measurements["major_diameter_mm"] - nominal_diameter) / DIAMETER_TOLERANCE_MM
    score = diameter_error ** 2

    if pitch_result.get("pitch_trusted"):
        pitch_error = (pitch_result["pitch_mm"] - nominal_pitch) / PITCH_TOLERANCE_MM
        score += pitch_error ** 2

    head_width = measurements.get("head_side_width_mm")
    if head_width and measurements.get("head_type") in ("hex", "socket"):
        closest_head_error = min(min(abs(head_width - flats), abs(head_width - flats * CORNERS_TO_FLATS_RATIO))
                                 for flats in wrench_sizes)
        score += 0.5 * (closest_head_error / HEAD_TOLERANCE_MM) ** 2
    return score


def snap_length(measured_length_mm, thread_system):
    """Nearest catalogue length; flags non-standard if more than 2.5 mm away."""
    standard_lengths = INCH_STANDARD_LENGTHS_MM if thread_system == "UNC" else METRIC_STANDARD_LENGTHS_MM
    nearest_length = min(standard_lengths, key=lambda length: abs(length - measured_length_mm))
    is_standard = abs(nearest_length - measured_length_mm) <= 2.5
    return nearest_length, is_standard


def match_specification(measurements, pitch_result):
    """Return dict with ranked candidates (top 3), probabilities and an ambiguity flag."""
    scored = []
    for spec in FASTENER_SPECS:
        scored.append((score_candidate(spec, measurements, pitch_result), spec))
    scored.sort(key=lambda item: item[0])

    weights = [math.exp(-score / 2) for score, _spec in scored]
    total_weight = sum(weights) + 1e-12
    candidates = []
    for (score, spec), weight in list(zip(scored, weights))[:3]:
        designation, thread_system, diameter, pitch, wrench_sizes = spec
        length, length_is_standard = snap_length(measurements["length_under_head_mm"], thread_system)
        candidates.append({"designation": designation, "thread_system": thread_system,
                           "nominal_diameter_mm": diameter, "nominal_pitch_mm": pitch,
                           "wrench_size_mm": wrench_sizes[0], "length_mm": length,
                           "length_is_standard": length_is_standard,
                           "probability": round(weight / total_weight, 3), "score": round(score, 2)})
    is_ambiguous = candidates[0]["probability"] < AMBIGUOUS_IF_PROBABILITY_BELOW
    return {"candidates": candidates, "is_ambiguous": is_ambiguous}
