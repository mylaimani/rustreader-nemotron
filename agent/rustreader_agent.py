"""
RustReader agent: a perception -> decision -> action loop around OpenCV 5 tools.

Every photo goes through a fixed list of OpenCV tools, but WHAT HAPPENS NEXT
depends on what the tools see:
  * markers missing / photo blurry / too far / two bolts  -> ask for a new photo
  * rust is moderate or heavy, or the plain edge is weak  -> re-run pitch with
                                                             the rust-enhanced edge tool
  * pitch still unreadable after one photo                -> ask for a closer photo
  * two specs still close                                 -> ask the human to confirm
  * confident                                             -> identify, give rust advice,
                                                             find the part (store / online)

Every decision is written to session["trace"] so judges (and users) can see
exactly which visual evidence changed which step.

When NEBIUS_API_KEY is set, NVIDIA Nemotron models on Nebius Token Factory
join the loop (agent/nemotron_review.py): Super VL cross-checks head type and
rust from the photo and writes the final explanation in English and Tamil
(Lightning is the fast fallback). Without the key the agent runs pure-OpenCV
exactly as before.
"""
import time
import uuid

from agent import nemotron_review
from vision import marker_detection, image_quality, segmentation, measurement
from vision import thread_pitch, rust_analysis, spec_matching

MINIMUM_MARKERS = 2
MINIMUM_PHOTO_PIXELS_PER_MM = 6.0
MAXIMUM_REPROJECTION_ERROR_MM = 0.5
PITCH_TRUSTED_CONFIDENCE = 12.0   # calibrate on real photos (eval/evaluate.py)
MAXIMUM_PHOTOS_BEFORE_ASKING_HUMAN = 2


def start_new_session():
    """Fresh agent state. Stored between requests (S3 in the cloud, a dict locally)."""
    return {"session_id": uuid.uuid4().hex[:12], "photo_count": 0, "status": "waiting_for_photo",
            "measurement_history": [], "pending_question": None, "final_result": None, "trace": []}


def add_trace_step(session, tool, observation, decision, reason):
    """Record one perception -> decision step."""
    session["trace"].append({"step": len(session["trace"]) + 1, "photo": session["photo_count"],
                             "tool": tool, "observation": observation, "decision": decision,
                             "reason": reason, "timestamp": round(time.time(), 3)})


def respond_request_new_photo(session, reason_code, message):
    session["status"] = "waiting_for_photo"
    return {"action": "request_new_photo", "reason_code": reason_code, "message": message}


def respond_ask_human(session, question, options):
    session["status"] = "waiting_for_human"
    session["pending_question"] = {"question": question, "options": options}
    return {"action": "request_human_confirmation", "question": question, "options": options}


# ── Perception steps. Each returns (result, response). A response means "stop here
#    and send this to the user"; None means "carry on". ──────────────────────────

def perceive_mat(session, photo_bgr):
    """Find markers, straighten the mat, check sharpness. Returns (rectified, response)."""
    found_markers = marker_detection.detect_mat_markers(photo_bgr)
    photo_pixels_per_mm = marker_detection.estimate_photo_pixels_per_mm(found_markers)
    observation = {"markers_found": sorted(found_markers), "photo_pixels_per_mm": round(photo_pixels_per_mm, 2)}
    if len(found_markers) < MINIMUM_MARKERS:
        add_trace_step(session, "detect_mat_markers", observation, "request_new_photo",
                       f"only {len(found_markers)} of 4 mat markers visible; cannot set scale")
        return None, respond_request_new_photo(session, "markers_missing",
                                               "I can't see the measuring mat's corner squares. "
                                               "Place the bolt on the mat and keep all 4 black squares in the photo.")
    if photo_pixels_per_mm < MINIMUM_PHOTO_PIXELS_PER_MM:
        add_trace_step(session, "detect_mat_markers", observation, "request_new_photo",
                       f"{photo_pixels_per_mm:.1f} px/mm is too coarse to see threads")
        return None, respond_request_new_photo(session, "too_far",
                                               "The photo is taken from too far away. Move closer so the mat fills the frame.")
    add_trace_step(session, "detect_mat_markers", observation, "continue",
                   "enough markers to compute scale and perspective")
    return straighten_and_check_quality(session, photo_bgr, found_markers)


def straighten_and_check_quality(session, photo_bgr, found_markers):
    """Homography + rectification + blur/brightness check. Returns (rectified, response)."""
    homography, reprojection_error_mm = marker_detection.compute_photo_to_mat_homography(found_markers)
    # Allow at least 3 photo pixels of corner error: on smaller photos one pixel is a bigger share of a mm
    photo_pixels_per_mm = marker_detection.estimate_photo_pixels_per_mm(found_markers)
    allowed_error_mm = max(MAXIMUM_REPROJECTION_ERROR_MM, 3.0 / max(photo_pixels_per_mm, 1.0))
    if homography is None or reprojection_error_mm > allowed_error_mm:
        add_trace_step(session, "compute_homography", {"reprojection_error_mm": reprojection_error_mm},
                       "request_new_photo", "markers do not agree on one flat sheet (mat bent or folded?)")
        return None, respond_request_new_photo(session, "mat_not_flat",
                                               "The mat doesn't look flat. Lay it flat on a table and try again.")
    rectified = marker_detection.rectify_photo_to_mat(photo_bgr, homography)
    quality = image_quality.assess_image_quality(rectified, photo_pixels_per_mm)
    quality["reprojection_error_mm"] = round(reprojection_error_mm, 3)
    quality["markers_used"] = len(found_markers)
    if "blurry" in quality["problems"]:
        add_trace_step(session, "assess_image_quality", quality, "request_new_photo",
                       "marker edges are soft, so thread edges will be unreliable")
        return None, respond_request_new_photo(session, "blurry",
                                               "The photo is blurry. Hold the phone steady (rest your elbows) and tap to focus on the bolt.")
    if "too_dark" in quality["problems"]:
        add_trace_step(session, "assess_image_quality", quality, "request_new_photo", "paper is too dark")
        return None, respond_request_new_photo(session, "too_dark",
                                               "The photo is too dark. Move to daylight or switch on a room light (no flash).")
    add_trace_step(session, "assess_image_quality", quality, "continue", "sharp and bright enough to measure")
    return rectified, None


def perceive_objects(session, rectified):
    """Find exactly one fastener on the mat. Returns (contour, response)."""
    contours, touches_border = segmentation.find_fastener_contours(rectified)
    observation = {"objects_found": len(contours), "largest_touches_border": touches_border}
    if len(contours) == 0:
        add_trace_step(session, "find_fastener_contours", observation, "request_new_photo", "mat is empty")
        return None, respond_request_new_photo(session, "no_object",
                                               "I don't see a bolt on the mat. Place one inside the dashed area.")
    if len(contours) > 1:
        add_trace_step(session, "find_fastener_contours", observation, "request_new_photo",
                       "more than one object; measurements would mix")
        return None, respond_request_new_photo(session, "multiple_objects",
                                               f"I see {len(contours)} objects. Leave only ONE bolt on the mat.")
    if touches_border:
        add_trace_step(session, "find_fastener_contours", observation, "request_new_photo",
                       "bolt crosses the work-area edge; its length would be cut short")
        return None, respond_request_new_photo(session, "object_outside_area",
                                               "Part of the bolt is outside the dashed area. Move it to the middle.")
    add_trace_step(session, "find_fastener_contours", observation, "continue", "exactly one fastener")
    return contours[0], None


def perceive_shape(session, rectified, contour):
    """Measure the outline. Returns ((measurements, aligned_bgr, aligned_mask), response)."""
    measurements, aligned_bgr, aligned_mask = measurement.measure_bolt(rectified, contour)
    observation = {key: value for key, value in measurements.items() if not key.endswith("_column")}
    if measurements["object_type"] == "nut_or_washer":
        add_trace_step(session, "measure_bolt", observation, "stop_unsupported",
                       "short and wide: looks like a nut or washer, which v1 does not measure")
        session["status"] = "finished"
        response = {"action": "unsupported_object",
                    "message": "This looks like a nut or washer. This version measures bolts and screws only."}
        return None, response
    add_trace_step(session, "measure_bolt", observation, "continue",
                   f"{measurements['object_type']} with shank about {measurements['major_diameter_mm']} mm")
    return (measurements, aligned_bgr, aligned_mask), None


# ── Decisions driven by what the tools saw ──────────────────────────────────────

def choose_and_run_pitch_tool(session, aligned_bgr, aligned_mask, measurements, rust):
    """Pick the edge tool from the rust level; retry with the other tool if weak."""
    start_column = measurements["shank_start_column"]
    end_column = measurements["shank_end_column"]
    rust_hides_edges = rust["rust_level"] in ("moderate", "heavy")
    first_mode = "enhanced" if rust_hides_edges else "mask"
    first_result = thread_pitch.estimate_thread_pitch(aligned_bgr, aligned_mask, start_column, end_column, first_mode,
                                                       diameter_mm=measurements["major_diameter_mm"])
    if first_result["confidence"] >= PITCH_TRUSTED_CONFIDENCE:
        reason = f"rust is {rust['rust_level']}, so started with the {first_mode} edge tool; peak is clear"
        add_trace_step(session, "estimate_thread_pitch", first_result, "accept_pitch", reason)
        first_result["pitch_trusted"] = True
        return first_result

    second_mode = "mask" if first_mode == "enhanced" else "enhanced"
    add_trace_step(session, "estimate_thread_pitch", first_result, f"retry_with_{second_mode}_edges",
                   f"FFT peak confidence {first_result['confidence']} is below {PITCH_TRUSTED_CONFIDENCE}")
    second_result = thread_pitch.estimate_thread_pitch(aligned_bgr, aligned_mask, start_column, end_column, second_mode,
                                                        diameter_mm=measurements["major_diameter_mm"])
    best_result = max(first_result, second_result, key=lambda result: result["confidence"])
    best_result["pitch_trusted"] = best_result["confidence"] >= PITCH_TRUSTED_CONFIDENCE
    decision = "accept_pitch" if best_result["pitch_trusted"] else "pitch_unreadable"
    add_trace_step(session, "estimate_thread_pitch", second_result, decision,
                   f"best of both edge tools: {best_result['edge_mode']} with confidence {best_result['confidence']}")
    return best_result


def fuse_with_previous_photos(session, measurements, pitch_result):
    """Combine this photo with earlier photos of the same bolt (median sizes, best pitch)."""
    session["measurement_history"].append({"measurements": measurements, "pitch": pitch_result})
    history = session["measurement_history"]
    fused = dict(measurements)
    for key in ("major_diameter_mm", "length_under_head_mm", "head_side_width_mm"):
        values = sorted(entry["measurements"][key] for entry in history if entry["measurements"].get(key))
        if values:
            fused[key] = values[len(values) // 2]
    best_pitch = max((entry["pitch"] for entry in history), key=lambda pitch: pitch["confidence"])
    if len(history) > 1:
        add_trace_step(session, "fuse_photos", {"photos_fused": len(history), "major_diameter_mm": fused["major_diameter_mm"],
                                                "best_pitch_mm": best_pitch["pitch_mm"]},
                       "continue", "combined measurements from all photos of this bolt")
    return fused, best_pitch


def decide_identification(session, measurements, pitch_result, rust):
    """Match specs, then identify, ask for a closer photo, or ask the human."""
    match = spec_matching.match_specification(measurements, pitch_result)
    top_candidates = match["candidates"]
    observation = {"top_candidates": [(c["designation"], c["probability"]) for c in top_candidates]}
    pitches_differ = len({c["nominal_pitch_mm"] for c in top_candidates[:2]}) > 1

    if match["is_ambiguous"] and not pitch_result["pitch_trusted"] and pitches_differ \
            and session["photo_count"] < MAXIMUM_PHOTOS_BEFORE_ASKING_HUMAN:
        add_trace_step(session, "match_specification", observation, "request_new_photo",
                       "top specs differ only in pitch and pitch was unreadable; a closer view can settle it")
        return respond_request_new_photo(session, "threads_unreadable",
                                         "I can tell the diameter but not the thread pitch. Take a closer photo "
                                         "(still showing all 4 corner squares) with light from the side.")
    if match["is_ambiguous"]:
        add_trace_step(session, "match_specification", observation, "request_human_confirmation",
                       "two or more specs fit almost equally well; a person should decide")
        options = [c["designation"] for c in top_candidates] + ["None of these"]
        session["pending_candidates"] = top_candidates
        return respond_ask_human(session, "These sizes fit almost equally. Which one is it? "
                                 "(Tip: try a nut you know, or a pitch gauge.)", options)
    add_trace_step(session, "match_specification", observation, "identify",
                   f"{top_candidates[0]['designation']} is clearly the best fit")
    return finish_identification(session, top_candidates[0], measurements, pitch_result, rust, confirmed_by_human=False)


# ── Actions ─────────────────────────────────────────────────────────────────────

RUST_ADVICE = {
    "none": "No visible rust. Reuse is usually fine if threads are undamaged.",
    "light": "Light surface rust. Wire-brush and inspect the threads before reuse.",
    "moderate": "Moderate rust. Replace if it goes into brakes, suspension, wheels, engine or "
                "any load-bearing joint; otherwise clean and inspect.",
    "heavy": "Heavy rust. Replace it. Do not reuse on any safety-critical joint.",
    "unknown": "Rust level could not be judged from this photo.",
}
SAFETY_NOTE = ("Advisory only: a photo cannot see stretch, fatigue or internal cracks. "
               "Torque-to-yield bolts are single-use regardless. When in doubt, replace.")


def build_find_part_actions(part_name):
    """Links the app would open next: nearby hardware shops, then online supplier."""
    search_text = part_name.replace(" ", "+")
    return [{"type": "find_nearby_store",
             "label": "Find in nearby hardware shops",
             "url": f"https://www.google.com/maps/search/{search_text}+hardware+shop"},
            {"type": "order_online", "label": "Order from The Industrial 360",
             "url": "https://theindustrial360.com"}]


def format_length_label(candidate):
    """Metric lengths in mm ("40"); inch bolts in inches ("1-1/2\"")."""
    if candidate["thread_system"] != "UNC":
        return str(candidate["length_mm"])
    eighths = round(candidate["length_mm"] / 25.4 * 8)
    whole_inches, remaining_eighths = divmod(eighths, 8)
    if remaining_eighths == 0:
        return f'{whole_inches}"'
    numerator, denominator = remaining_eighths, 8
    while numerator % 2 == 0:
        numerator, denominator = numerator // 2, denominator // 2
    fraction = f"{numerator}/{denominator}"
    return f'{whole_inches}-{fraction}"' if whole_inches else f'{fraction}"'


def finish_identification(session, candidate, measurements, pitch_result, rust, confirmed_by_human):
    """Final answer: spec, size, rust advice and what to do next."""
    head_type = measurements.get("head_type", "hex")
    part_name = f"{candidate['designation']} x {format_length_label(candidate)} {spec_matching.PART_NAMES[head_type]}"
    tool = spec_matching.describe_tool(candidate["designation"], head_type, candidate["wrench_size_mm"])
    replace_recommended = rust["rust_level"] in ("moderate", "heavy")
    result = {"part_name": part_name, "designation": candidate["designation"],
              "thread_system": candidate["thread_system"], "length_mm": candidate["length_mm"],
              "length_is_standard": candidate["length_is_standard"],
              "wrench_size_mm": candidate["wrench_size_mm"], "tool": tool, "head_type": head_type,
              "probability": candidate["probability"],
              "confirmed_by_human": confirmed_by_human,
              "measured": {"major_diameter_mm": measurements["major_diameter_mm"],
                           "length_under_head_mm": measurements["length_under_head_mm"],
                           "head_side_width_mm": measurements.get("head_side_width_mm"),
                           "pitch_mm": pitch_result["pitch_mm"], "pitch_trusted": pitch_result["pitch_trusted"]},
              "rust": rust, "rust_advice": RUST_ADVICE[rust["rust_level"]], "safety_note": SAFETY_NOTE,
              "replace_recommended": replace_recommended,
              "next_actions": build_find_part_actions(part_name)}
    explanation = nemotron_review.explain_identification(result, session.get("nemotron_vision"))
    if explanation:
        result["explanation"] = explanation
        add_trace_step(session, "nemotron_explain", {"model": explanation["model"], "languages": ["en", "ta"]},
                       "attach_explanation",
                       f"NVIDIA {explanation['model'].split('/')[-1]} wrote the mechanic-friendly explanation "
                       "in English and Tamil")
    decision = "recommend_replacement_and_find_part" if replace_recommended else "identify_and_offer_part_search"
    add_trace_step(session, "plan_next_action", {"rust_level": rust["rust_level"]}, decision,
                   "rust level decides whether to push replacement or just offer the part")
    session["status"] = "finished"
    session["final_result"] = result
    return {"action": "final_identification", "result": result}


def answer_human_question(session, chosen_option):
    """Called when the user picks one of the offered sizes."""
    candidates = session.get("pending_candidates") or []
    chosen = next((c for c in candidates if c["designation"] == chosen_option), None)
    add_trace_step(session, "human_confirmation", {"chosen": chosen_option},
                   "identify" if chosen else "stop_no_match", "human made the final call")
    session["pending_question"] = None
    if chosen is None:
        session["status"] = "finished"
        return {"action": "no_match", "message": "Thanks. This one will be added to our review list."}
    last_entry = session["measurement_history"][-1]
    return finish_identification(session, chosen, last_entry["measurements"], last_entry["pitch"],
                                 session["last_rust"], confirmed_by_human=True)


# ── Entry point ─────────────────────────────────────────────────────────────────

def run_agent_on_photo(session, photo_bgr):
    """One turn of the loop for one uploaded photo. Returns (response, visuals)."""
    session["photo_count"] += 1
    rectified, response = perceive_mat(session, photo_bgr)
    if response:
        return response, {}
    contour, response = perceive_objects(session, rectified)
    if response:
        return response, {}
    shape, response = perceive_shape(session, rectified, contour)
    if response:
        return response, {}
    measurements, aligned_bgr, aligned_mask = shape

    rust = rust_analysis.measure_rust(aligned_bgr, aligned_mask)
    session["last_rust"] = rust
    add_trace_step(session, "measure_rust", rust, "choose_pitch_tool",
                   "rust level decides which edge tool to try first")
    vision_opinion = nemotron_review.vision_second_opinion(aligned_bgr, measurements, rust)
    if vision_opinion:
        session["nemotron_vision"] = vision_opinion
        add_trace_step(session, "nemotron_vision_check", vision_opinion, "continue",
                       "NVIDIA Nemotron Super VL (Nebius Token Factory) cross-checked head type and rust")
    pitch_result = choose_and_run_pitch_tool(session, aligned_bgr, aligned_mask, measurements, rust)
    fused_measurements, best_pitch = fuse_with_previous_photos(session, measurements, pitch_result)
    response = decide_identification(session, fused_measurements, best_pitch, rust)
    visuals = {"aligned": aligned_bgr, "aligned_mask": aligned_mask, "measurements": measurements}
    return response, visuals
