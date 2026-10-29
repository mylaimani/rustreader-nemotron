# Demo video (max 3 minutes) — shot list for the Nebius x NVIDIA hackathon

Record screen + phone. Upload to YouTube as **Public** (the rules say public, not unlisted).
Judges must see: the project working, how Nemotron is used, and that it runs on Nebius Token Factory.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:20 | You, holding a rusted bolt | "I'm Mani, a software developer in Tamil Nadu. This ₹30 bolt nearly cost a ₹5,000 repair because nobody could read it. RustReader identifies rusted bolts from one photo — and explains the answer in the mechanic's own language." |
| 0:20–0:40 | Printed mat, phone, one bolt placed | "Print this A4 mat. The four markers give the scale. One bolt, one photo." |
| 0:40–1:20 | Phone: open the live URL, scan a real rusty bolt, result page | Read out: part name, pitch, length, rust level, spanner size. Then scroll to the **"Nemotron explains"** panel: read the English line, show the Tamil line. "That explanation is written by NVIDIA Nemotron-3.5-Super-VL, served by Nebius Token Factory." |
| 1:20–1:50 | Decision trace on screen | Point at the two Nemotron steps: **nemotron_vision_check** — "Nemotron looked at the same photo on its own and said hex head, heavy rust — it agrees with OpenCV, and the trace records that" — and **nemotron_explain**. |
| 1:50–2:20 | Code: `agent/nemotron_client.py` on screen (the payload with `chat_template_kwargs` and the Token Factory URL) | "Two NVIDIA open models on Token Factory: Super VL for vision and the Tamil explanation, Lightning as a fast fallback. Plain HTTPS, no SDK. One thing we learned: send `enable_thinking: false`, otherwise the reasoning model spends its whole budget thinking and returns nothing." |
| 2:20–2:40 | Nebius Token Factory console: the two models, usage/credits page | "Both models run on Nebius Token Factory. A full scan — one vision call, one text call — costs about a tenth of a cent." |
| 2:40–3:00 | Back to the phone, nearby-shops button; end card with repo URL | "Measurement by OpenCV, judgement and explanation by Nemotron, delivered to a mechanic in Tamil. Code is MIT on GitHub. Thanks." |

Before recording:
- Run `bash infra/smoke_test.sh <live URL>` once so the Lambda is warm and you have confirmed the Nemotron steps appear.
- Have `samples/synthetic/M12x1.25x50_heavy.jpg` ready as a backup if the live bolt photo misbehaves.
- Keep the Token Factory API key off screen (close the API keys page before recording).
