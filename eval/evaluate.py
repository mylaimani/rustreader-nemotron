"""
Evaluates the agent on labelled photos and writes a results table + summary.

    python eval/evaluate.py eval/ground_truth.csv            (your real photos)
    python eval/evaluate.py eval/synthetic_ground_truth.csv  (synthetic set)

photo_file paths are relative to the CSV file (or bare names inside samples/).
Each photo is ONE first-turn scan in a fresh session: we record what the agent
did (identify / ask for new photo / ask human) and, when it measured, the errors.
"""
import csv
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import rustreader_agent  # noqa: E402

SAMPLES_FOLDER = os.path.join(os.path.dirname(__file__), "..", "samples")


def resolve_photo_path(csv_path, photo_file):
    beside_csv = os.path.join(os.path.dirname(csv_path), photo_file)
    if os.path.exists(beside_csv):
        return beside_csv
    return os.path.join(SAMPLES_FOLDER, photo_file)


def to_number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def evaluate_one_photo(photo_path, truth):
    """Run one fresh session on the photo and compare with the ground truth row."""
    session = rustreader_agent.start_new_session()
    started = time.time()
    response, _visuals = rustreader_agent.run_agent_on_photo(session, cv2.imread(photo_path))
    row = {"photo_file": truth["photo_file"], "true_designation": truth["designation"],
           "action": response["action"], "milliseconds": round((time.time() - started) * 1000),
           "trace_steps": len(session["trace"]), "notes": truth.get("notes", "")}
    history = session["measurement_history"]
    if not history:
        return row
    measured = history[-1]["measurements"]
    pitch = history[-1]["pitch"]
    row["measured_diameter_mm"] = measured.get("major_diameter_mm")
    row["measured_length_mm"] = measured.get("length_under_head_mm")
    row["measured_pitch_mm"] = pitch.get("pitch_mm") if pitch.get("pitch_trusted") else None
    row["pitch_confidence"] = pitch.get("confidence")
    add_errors(row, truth)
    if response["action"] == "final_identification":
        row["predicted_designation"] = response["result"]["designation"]
        row["designation_correct"] = response["result"]["designation"] == truth["designation"]
    return row


def add_errors(row, truth):
    """Absolute errors in mm against caliper/pitch-gauge values."""
    pairs = [("diameter_error_mm", "measured_diameter_mm", "diameter_mm"),
             ("length_error_mm", "measured_length_mm", "length_mm"),
             ("pitch_error_mm", "measured_pitch_mm", "pitch_mm")]
    for error_key, measured_key, truth_key in pairs:
        measured_value = row.get(measured_key)
        true_value = to_number(truth.get(truth_key))
        if measured_value is not None and true_value is not None:
            row[error_key] = round(abs(measured_value - true_value), 3)


def summarise(rows):
    """Text summary: counts per action, accuracy, mean/max errors, failure list."""
    lines = [f"Photos evaluated: {len(rows)}"]
    for action in sorted({row["action"] for row in rows}):
        lines.append(f"  {action}: {sum(1 for row in rows if row['action'] == action)}")
    identified = [row for row in rows if row["action"] == "final_identification"]
    if identified:
        correct = sum(1 for row in identified if row.get("designation_correct"))
        lines.append(f"Correct thread spec when identified: {correct}/{len(identified)} "
                     f"({100 * correct / len(identified):.0f}%)")
    for error_key in ("diameter_error_mm", "length_error_mm", "pitch_error_mm"):
        errors = [row[error_key] for row in rows if row.get(error_key) is not None]
        if errors:
            lines.append(f"{error_key}: mean {statistics.mean(errors):.2f}, max {max(errors):.2f} (n={len(errors)})")
    lines.append(f"Median processing time: {statistics.median(row['milliseconds'] for row in rows)} ms")
    lines.append("Not identified on first photo / wrong:")
    for row in rows:
        if row["action"] != "final_identification" or not row.get("designation_correct"):
            lines.append(f"  - {row['photo_file']}: {row['action']} {row.get('predicted_designation', '')} | {row['notes']}")
    return "\n".join(lines)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "ground_truth.csv")
    with open(csv_path, newline="") as file:
        truth_rows = list(csv.DictReader(file))
    results = []
    for truth in truth_rows:
        photo_path = resolve_photo_path(csv_path, truth["photo_file"])
        if not os.path.exists(photo_path):
            print("missing photo, skipped:", photo_path)
            continue
        results.append(evaluate_one_photo(photo_path, truth))
    output_base = os.path.splitext(csv_path)[0] + "_results"
    columns = sorted({key for row in results for key in row})
    with open(output_base + ".csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    summary = summarise(results)
    with open(output_base + "_summary.txt", "w") as file:
        file.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
