"""
The two places where NVIDIA Nemotron (served by Nebius Token Factory) joins
the agent loop:

  1. vision_second_opinion  - Nemotron-3.5-Super-VL looks at the straightened
     bolt photo and independently names the head type and rust level, so the
     OpenCV measurement is cross-checked by a second, different kind of eye.
  2. explain_identification - Nemotron-3.5-Super-VL (text only) turns the final
     numeric result into two or three plain sentences for the mechanic, in
     English and in Tamil; Nemotron-3.5-Lightning is the fast fallback.

Both return None when NEBIUS_API_KEY is not set or a call fails, and the
agent then behaves exactly like the pure-OpenCV version.
"""
import base64
import json

import cv2

from agent import nemotron_client

VISION_SYSTEM_PROMPT = ("detailed thinking off. "
                        "You inspect fasteners (bolts, screws, nuts) in photos for a workshop. "
                        "Reply with ONE JSON object only, no other text.")
EXPLAIN_SYSTEM_PROMPT = ("detailed thinking off. "
                         "You explain bolt identification results to workshop mechanics in simple words. "
                         "Reply with ONE JSON object only, no other text.")


def encode_jpeg_for_model(aligned_bgr, maximum_width=640):
    """Small JPEG so the vision call stays fast and cheap."""
    if aligned_bgr.shape[1] > maximum_width:
        scale = maximum_width / aligned_bgr.shape[1]
        aligned_bgr = cv2.resize(aligned_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _ok, encoded = cv2.imencode(".jpg", aligned_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def vision_second_opinion(aligned_bgr, measurements, rust):
    """Ask Super VL what it sees and compare with OpenCV. None when unavailable."""
    if not nemotron_client.nemotron_is_configured():
        return None
    question = ("This straightened photo shows one fastener lying on white paper. "
                "Look at it and reply with exactly this JSON shape, filled in with what YOU see: "
                '{"head_type": "<one of: hex, socket, countersunk, flanged, unknown>", '
                '"rust_level": "<one of: none, light, moderate, heavy>", '
                '"note": "<one short sentence about its condition>"}')
    reply = nemotron_client.ask_nemotron_about_image(VISION_SYSTEM_PROMPT, question,
                                                     encode_jpeg_for_model(aligned_bgr))
    opinion = nemotron_client.parse_json_reply(reply)
    if opinion is None:
        return None
    if opinion.get("head_type") not in ("hex", "socket", "countersunk", "flanged"):
        opinion["head_type"] = "unknown"
    if opinion.get("rust_level") not in ("none", "light", "moderate", "heavy"):
        opinion["rust_level"] = "unknown"
    return {"model": nemotron_client.VISION_MODEL,
            "head_type_seen": str(opinion.get("head_type", "unknown")),
            "rust_level_seen": str(opinion.get("rust_level", "unknown")),
            "note": str(opinion.get("note", ""))[:200],
            "agrees_with_opencv_head": opinion.get("head_type") == measurements.get("head_type"),
            "agrees_with_opencv_rust": opinion.get("rust_level") == rust.get("rust_level")}


def explain_identification(result, vision_opinion):
    """2-3 plain sentences about the final result, in English and Tamil. None when unavailable."""
    if not nemotron_client.nemotron_is_configured():
        return None
    facts = {"part": result["part_name"], "thread_system": result["thread_system"],
             "spanner_or_tool": result["tool"],
             "measured_diameter_mm": result["measured"]["major_diameter_mm"],
             "measured_length_mm": result["measured"]["length_under_head_mm"],
             "thread_pitch_mm": result["measured"]["pitch_mm"],
             "rust_level": result["rust"]["rust_level"],
             "replace_recommended": result["replace_recommended"],
             "second_opinion_note": (vision_opinion or {}).get("note", "")}
    question = ("A photo-measurement app identified this fastener. The facts:\n" + json.dumps(facts) +
                "\n\nWrite 2-3 short sentences for the mechanic who owns it: what the part is, "
                "which spanner or tool fits it, and what to do about the rust. "
                "Then translate your sentences into simple spoken Tamil (Tamil script), keeping "
                "sizes like M12 x 1.25 and tool names in English.\n"
                "Reply with exactly this JSON shape, filled in with your own words: "
                '{"english": "<your English sentences>", "tamil": "<your Tamil sentences>"}')
    return ask_for_explanation(question)


def looks_like_real_explanation(explanation):
    """False when the model echoed the template instead of writing real sentences."""
    if explanation is None or not explanation.get("english"):
        return False
    english = str(explanation["english"])
    return "<" not in english and "your English" not in english


def ask_for_explanation(question):
    """Try the explainer model (Super VL) first; fall back to Lightning so the demo never goes blank."""
    model_used = nemotron_client.EXPLAIN_MODEL
    reply = nemotron_client.ask_nemotron_text(EXPLAIN_SYSTEM_PROMPT, question)
    explanation = nemotron_client.parse_json_reply(reply)
    if not looks_like_real_explanation(explanation):
        model_used = nemotron_client.TEXT_MODEL
        reply = nemotron_client.ask_nemotron_text(EXPLAIN_SYSTEM_PROMPT, question, model_name=model_used)
        explanation = nemotron_client.parse_json_reply(reply)
    if not looks_like_real_explanation(explanation):
        return None
    return {"model": model_used,
            "english": str(explanation.get("english", "")),
            "tamil": str(explanation.get("tamil", ""))}
