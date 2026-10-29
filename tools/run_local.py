"""
Run the agent on photos from the command line (no AWS needed).

    python tools/run_local.py photo1.jpg [photo2.jpg ...] [--answer "M8"]

Photos are treated as successive turns of ONE session (like a user retaking
a photo when asked). The full decision trace is printed at the end.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import rustreader_agent  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--answer", help="answer to give if the agent asks a human")
    parser.add_argument("--trace", action="store_true", help="print the full decision trace")
    arguments = parser.parse_args()

    session = rustreader_agent.start_new_session()
    for photo_path in arguments.photos:
        photo = cv2.imread(photo_path)
        response, _visuals = rustreader_agent.run_agent_on_photo(session, photo)
        print(f"\n[{os.path.basename(photo_path)}] -> {response['action']}")
        print(json.dumps(response, indent=2, default=str)[:1500])
        if response["action"] == "request_human_confirmation" and arguments.answer:
            response = rustreader_agent.answer_human_question(session, arguments.answer)
            print(f"\n[human answered {arguments.answer}] -> {response['action']}")
        if session["status"] == "finished":
            break
    if arguments.trace:
        print("\nDECISION TRACE")
        for step in session["trace"]:
            print(f"  {step['step']:>2}. photo {step['photo']} | {step['tool']:<24} -> {step['decision']:<34} | {step['reason']}")


if __name__ == "__main__":
    main()
