#!/usr/bin/env bash
# Sends the bundled heavy-rust sample to the deployed endpoint and prints the agent's answer,
# including the NVIDIA Nemotron steps when NEBIUS_API_KEY is set on the Lambda.
#   bash infra/smoke_test.sh https://xxxx.lambda-url.ap-southeast-2.on.aws/
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
result = body["response"].get("result", {})
print("part:", result.get("part_name"))
print("trace steps:", len(body["trace"]))
nemotron_step_names = [step.get("tool", "") for step in body["trace"]]
nemotron_step_names = [name for name in nemotron_step_names if "nemotron" in name]
print("nemotron trace steps:", nemotron_step_names or "NONE - is NEBIUS_API_KEY set on the Lambda?")
explanation = result.get("explanation")
if explanation:
    print("explained by:", explanation.get("model"))
    print("english:", explanation.get("english"))
    print("tamil:", explanation.get("tamil"))
PY
