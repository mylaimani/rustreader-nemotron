"""
Live test of the Nemotron integration. Run on your own machine (needs internet):

    cd rustreader_cloud
    export NEBIUS_API_KEY="...your Token Factory key..."
    python3 tools/test_nemotron.py samples/synthetic/M12x1.25x50_heavy.jpg

It runs one full agent turn on the photo and prints the Nemotron vision
second opinion, the English/Tamil explanation and the decision trace.
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import nemotron_client, rustreader_agent  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("usage: python3 tools/test_nemotron.py <photo.jpg>")
        return 1
    if not nemotron_client.nemotron_is_configured():
        print("NEBIUS_API_KEY is not set. Export it first (see the docstring above).")
        return 1

    photo = cv2.imread(sys.argv[1])
    if photo is None:
        print(f"could not read {sys.argv[1]}")
        return 1
    print(f"models: vision={nemotron_client.VISION_MODEL}  explain={nemotron_client.EXPLAIN_MODEL}  "
          f"fallback={nemotron_client.TEXT_MODEL}")

    session = rustreader_agent.start_new_session()
    response, _visuals = rustreader_agent.run_agent_on_photo(session, photo)
    print(f"\nagent action: {response['action']}")

    vision_opinion = session.get("nemotron_vision")
    print("\nNEMOTRON VISION SECOND OPINION")
    print(json.dumps(vision_opinion, indent=2, ensure_ascii=False) if vision_opinion else "  (no reply - check key/model)")

    explanation = (session.get("final_result") or {}).get("explanation")
    print("\nNEMOTRON EXPLANATION")
    print(json.dumps(explanation, indent=2, ensure_ascii=False) if explanation else "  (no reply or no final result)")

    print("\nDECISION TRACE")
    for step in session["trace"]:
        print(f"  {step['step']:>2}. {step['tool']:<24} -> {step['decision']:<24} | {step['reason'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
