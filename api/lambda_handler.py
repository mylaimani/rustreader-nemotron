"""
AWS Lambda entry point (served through a Lambda Function URL).

Routes:
  GET  /                -> the web page (web/index.html)
  POST /analyze         -> {"session_id"?: str, "image_base64": str}   one agent turn
  POST /answer          -> {"session_id": str, "option": str}          human confirmation
  GET  /session/<id>    -> full session incl. decision trace

Each request logs CloudWatch Embedded Metric Format lines (latency, action),
so dashboards and alarms work without extra libraries.
"""
import base64
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import rustreader_agent  # noqa: E402
from api import session_store  # noqa: E402
from vision import annotation, rust_analysis  # noqa: E402

WEB_PAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "web", "index.html")
MAXIMUM_IMAGE_BYTES = 6_000_000


def convert_numpy_value(value):
    """json.dumps helper: numpy numbers -> plain Python numbers."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def make_response(status_code, body, content_type="application/json"):
    text = body if isinstance(body, str) else json.dumps(body, default=convert_numpy_value)
    return {"statusCode": status_code, "headers": {"Content-Type": content_type}, "body": text}


def read_json_body(event):
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)


def encode_jpeg_base64(image_bgr, maximum_width=900):
    """Small JPEG for the web page."""
    if image_bgr.shape[1] > maximum_width:
        scale = maximum_width / image_bgr.shape[1]
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def log_metrics(route, action, elapsed_milliseconds):
    """CloudWatch Embedded Metric Format: one JSON line becomes real metrics."""
    print(json.dumps({"_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
        "Namespace": "RustReader", "Dimensions": [["Route", "Action"]],
        "Metrics": [{"Name": "LatencyMs", "Unit": "Milliseconds"}, {"Name": "Requests", "Unit": "Count"}]}]},
        "Route": route, "Action": action, "LatencyMs": elapsed_milliseconds, "Requests": 1}))


def build_visuals_for_page(visuals):
    """Annotated bolt, rust overlay and Rust Mode view as base64 JPEGs."""
    if not visuals:
        return {}
    aligned_bgr = visuals["aligned"]
    aligned_mask = visuals["aligned_mask"]
    return {"measured": encode_jpeg_base64(annotation.draw_measurement_overlay(aligned_bgr, visuals["measurements"])),
            "rust_overlay": encode_jpeg_base64(rust_analysis.build_rust_overlay(aligned_bgr, aligned_mask)),
            "rust_mode": encode_jpeg_base64(rust_analysis.build_rust_mode_view(aligned_bgr))}


def handle_analyze(request):
    """One agent turn for one uploaded photo."""
    image_bytes = base64.b64decode(request["image_base64"].split(",")[-1])
    if len(image_bytes) > MAXIMUM_IMAGE_BYTES:
        return make_response(413, {"error": "Image too large; resize to under 6 MB."})
    photo_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if photo_bgr is None:
        return make_response(400, {"error": "Could not read the image."})

    session = session_store.load_session(request.get("session_id"))
    if session is None or session.get("status") == "finished":
        session = rustreader_agent.start_new_session()
    response, visuals = rustreader_agent.run_agent_on_photo(session, photo_bgr)
    session_store.save_scan_image(session["session_id"], session["photo_count"], photo_bgr, "original")
    session_store.save_session(session)
    body = {"session_id": session["session_id"], "photo_count": session["photo_count"],
            "response": response, "trace": session["trace"], "images": build_visuals_for_page(visuals)}
    return make_response(200, body)


def handle_answer(request):
    session = session_store.load_session(request.get("session_id"))
    if session is None or session.get("status") != "waiting_for_human":
        return make_response(400, {"error": "No question is waiting for this session."})
    response = rustreader_agent.answer_human_question(session, request.get("option", ""))
    session_store.save_session(session)
    return make_response(200, {"session_id": session["session_id"], "response": response, "trace": session["trace"]})


STATIC_FILES_FOLDER = os.path.join(os.path.dirname(__file__), "..")


def serve_static_file(path):
    """The printable mat and the demo sample photos (binary, base64-encoded for Lambda)."""
    if path == "/mat.pdf":
        file_path, content_type = os.path.join(STATIC_FILES_FOLDER, "docs", "RustReader_Measuring_Mat_A4.pdf"), "application/pdf"
    else:
        file_name = os.path.basename(path)
        file_path, content_type = os.path.join(STATIC_FILES_FOLDER, "samples", "synthetic", file_name), "image/jpeg"
    if not os.path.isfile(file_path):
        return make_response(404, {"error": "not found"})
    with open(file_path, "rb") as file:
        data = base64.b64encode(file.read()).decode("ascii")
    return {"statusCode": 200, "headers": {"Content-Type": content_type}, "body": data, "isBase64Encoded": True}


def lambda_handler(event, context):
    started = time.time()
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")
    action = "page"
    if method == "GET" and path == "/":
        with open(WEB_PAGE_PATH, encoding="utf-8") as file:
            result = make_response(200, file.read(), "text/html; charset=utf-8")
    elif method == "POST" and path == "/analyze":
        result = handle_analyze(read_json_body(event))
    elif method == "POST" and path == "/answer":
        result = handle_answer(read_json_body(event))
    elif method == "GET" and (path == "/mat.pdf" or path.startswith("/sample/")):
        result = serve_static_file(path)
    elif method == "GET" and path.startswith("/session/"):
        session = session_store.load_session(path.split("/")[-1])
        result = make_response(200 if session else 404, session or {"error": "not found"})
    else:
        result = make_response(404, {"error": "unknown route"})
    if result["statusCode"] == 200 and result["headers"]["Content-Type"].startswith("application/json"):
        action = json.loads(result["body"]).get("response", {}).get("action", "data")
    log_metrics(path.split("/")[1] or "page", action, round((time.time() - started) * 1000, 1))
    return result
