#!/usr/bin/env bash
# Sends the bundled heavy-rust sample to the deployed endpoint and prints the agent's answer.
#   bash infra/smoke_test.sh https://xxxx.lambda-url.ap-south-1.on.aws/
set -euo pipefail
URL="${1%/}"
python3 - "$URL" <<'PY'
import base64, json, sys, time, urllib.request
url = sys.argv[1]
image = base64.b64encode(open("samples/synthetic/M12x1.25x50_heavy.jpg", "rb").read()).decode()
request = urllib.request.Request(url + "/analyze", data=json.dumps({"image_base64": image}).encode(),
                                 headers={"Content-Type": "application/json"})
started = time.time()
body = json.loads(urllib.request.urlopen(request, timeout=60).read())
print("round trip ms:", round((time.time() - started) * 1000))
print("action:", body["response"]["action"])
print("part:", body["response"].get("result", {}).get("part_name"))
print("trace steps:", len(body["trace"]))
PY
