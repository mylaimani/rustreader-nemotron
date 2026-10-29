"""
Where sessions and scan images are kept.
  * In AWS (environment variable SCAN_BUCKET set): Amazon S3.
  * On a laptop: a local folder, so everything runs without AWS.
Images are re-encoded by OpenCV before saving, which strips EXIF (incl. GPS location).
"""
import json
import os

import cv2

BUCKET_NAME = os.environ.get("SCAN_BUCKET")
LOCAL_FOLDER = os.path.join(os.path.dirname(__file__), "..", ".local_store")


def convert_numpy_value(value):
    """json.dumps helper: numpy numbers -> plain Python numbers."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_bytes(key, data, content_type):
    if BUCKET_NAME:
        import boto3  # included in the AWS Lambda Python runtime
        boto3.client("s3").put_object(Bucket=BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
        return
    path = os.path.join(LOCAL_FOLDER, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        file.write(data)


def _read_bytes(key):
    if BUCKET_NAME:
        import boto3
        try:
            return boto3.client("s3").get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()
        except Exception:
            return None
    path = os.path.join(LOCAL_FOLDER, key)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as file:
        return file.read()


def save_session(session):
    _write_bytes(f"sessions/{session['session_id']}.json",
                 json.dumps(session, default=convert_numpy_value).encode("utf-8"), "application/json")


def load_session(session_id):
    """Return the stored session dict, or None if it does not exist."""
    if not session_id or not session_id.isalnum():
        return None
    data = _read_bytes(f"sessions/{session_id}.json")
    return json.loads(data) if data else None


def save_scan_image(session_id, photo_number, image_bgr, label):
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        _write_bytes(f"scans/{session_id}/photo{photo_number}_{label}.jpg", encoded.tobytes(), "image/jpeg")
