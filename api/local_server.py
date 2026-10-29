"""
Run the same Lambda code on your Mac for testing:

    python api/local_server.py      then open http://localhost:8080

It turns each browser request into a Lambda Function URL event.
"""
import base64
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.lambda_handler import lambda_handler  # noqa: E402

PORT = int(os.environ.get("PORT", "8080"))


class LambdaEmulatorHandler(BaseHTTPRequestHandler):
    def handle_any(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        event = {"requestContext": {"http": {"method": method}}, "rawPath": self.path.split("?")[0],
                 "body": base64.b64encode(body).decode("ascii"), "isBase64Encoded": True}
        result = lambda_handler(event, None)
        if result.get("isBase64Encoded"):
            payload = base64.b64decode(result["body"])
        else:
            payload = result["body"].encode("utf-8")
        self.send_response(result["statusCode"])
        self.send_header("Content-Type", result["headers"]["Content-Type"])
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.handle_any("GET")

    def do_POST(self):
        self.handle_any("POST")


if __name__ == "__main__":
    print(f"RustReader local server on http://localhost:{PORT}")
    HTTPServer(("0.0.0.0", PORT), LambdaEmulatorHandler).serve_forever()
