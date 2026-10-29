# Nebius x NVIDIA Global AI Hackathon — submission fields (copy/paste)

## Project name
RustReader — rusted bolt identification with NVIDIA Nemotron on Nebius

## Elevator pitch
Point your phone at a rusted bolt nobody can read. An OpenCV measurement agent finds its size, and NVIDIA Nemotron on Nebius Token Factory double-checks the photo and explains the result to the mechanic in English and Tamil.

## Track
Best Apps and Agents

## About the project

### Inspiration
Old fasteners rust, and rust hides everything: head markings, thread pitch, sometimes even the size. A relative of mine who works in fastener manufacturing could not identify a battery bolt worth about ₹30; the "closest fit" he used instead risked a ₹5,000 repair. Mechanics here travel 15 km to hardware shops carrying the old bolt to match it. Existing fastener-ID apps are trained on clean catalogue photos and fail on exactly these rusted parts.

### What it does
Place the bolt on a printed A4 mat (four ArUco markers) and take one photo. RustReader:
- straightens the photo and converts pixels to millimetres,
- measures shank diameter, length and head type,
- reads the thread pitch with an FFT of the thread edge, even under rust,
- scores rust severity,
- matches ISO metric coarse/fine and UNC tables, e.g. **M12x1.25 x 50 hex bolt, spanner 18 mm**,
- asks **NVIDIA Nemotron-3.5-Super-VL** to look at the same photo and independently name the head type and rust level — a second, different kind of eye on the measurement,
- asks Nemotron to write 2–3 plain sentences for the mechanic — what the part is, which spanner fits, what to do about the rust — in **English and Tamil**,
- recommends replace or reuse (advisory) and links to nearby hardware shops or an online supplier.

The agent loop decides what happens next from what the tools see: bad photo → ask for a new one; heavy rust → different edge tool; two sizes still fit → ask the human. Every step, including both Nemotron calls and whether Nemotron agreed with OpenCV, is recorded in a decision trace shown on the result page.

### How we used NVIDIA Nemotron and Nebius Token Factory
Two NVIDIA open models, both served by Nebius Token Factory through its OpenAI-compatible Chat Completions API (`agent/nemotron_client.py`, `agent/nemotron_review.py`):

| Model | Job |
|---|---|
| `nvidia/Nemotron-3.5-Super-VL` (120B, vision) | Second opinion on the straightened bolt photo (head type, rust level, one-line condition note), and the English + Tamil explainer |
| `nvidia/Nemotron-3_5-Lightning` (30B, 3B active) | Fast fallback explainer when Super VL does not return a usable answer |

Design choices:
- Calls are plain `urllib` POSTs — no SDK in the Lambda bundle, so the Token Factory integration is about 100 lines.
- Both calls send `chat_template_kwargs: {"enable_thinking": false}`. Without it these reasoning models spend the whole token budget thinking and return an empty answer.
- Every Nemotron call is optional and fails soft: without `NEBIUS_API_KEY` the agent runs pure-OpenCV exactly as before. The demo never goes blank because of a model timeout.
- Images are shrunk to 640 px JPEG before the vision call to keep it fast and cheap. One vision + one text call per identified bolt costs about $0.001, so the Builder Program's $25 credit covers roughly 25,000 scans.

Nebius: Token Factory for both models. Hosting is AWS Lambda (arm64) + S3, because the OpenCV part of the pipeline was already deployed there; the Nemotron calls are the only external service the Lambda talks to.

### What was built during the submission period
The OpenCV measurement pipeline existed before the hackathon (built for the OpenCV AI Competition, Oct 2026). New for this hackathon:
- `agent/nemotron_client.py` — Token Factory client (thinking switch, JSON-reply parser that survives `<think>` blocks and quoted example JSON, vision reply-as-list handling)
- `agent/nemotron_review.py` — vision second opinion + bilingual explanation, with Super VL → Lightning fallback
- Nemotron steps in the agent loop and decision trace (`agent/rustreader_agent.py`)
- "Nemotron explains" panel on the result page (`web/index.html`)
- `NEBIUS_API_KEY` wiring in `infra/deploy.sh`, `tools/test_nemotron.py`, MIT licence, this repo

### Challenges we ran into
- **Empty replies from reasoning models.** The first calls to Nemotron returned nothing: the model used the entire `max_tokens` on its hidden thinking. Fixed with `chat_template_kwargs.enable_thinking = false` plus "detailed thinking off" in the system prompt.
- **Tamil quality.** We tried Lightning (30B) as the main explainer because it is faster and cheaper. Its English was fine but its Tamil mixed in Hindi and Finnish words. The 120B Super VL writes clean spoken Tamil, so it took over the explainer job and Lightning became the fallback.
- **Template echo.** Occasionally a model returned our JSON template with the placeholders unfilled. `looks_like_real_explanation()` catches this and triggers the fallback model.
- **Parsing.** Reasoning models sometimes quote example JSON inside their thinking, so we parse the *last* JSON object in the reply, after any `</think>` tag.

### Accomplishments that we're proud of
- A mechanic who reads little technical English gets the answer in Tamil, on a phone, from one photo.
- Nemotron's independent head-type and rust reading is recorded next to the OpenCV measurement in the trace, so a disagreement is visible instead of silently averaged away.
- The whole Token Factory integration is small, dependency-free and switchable with one environment variable.

### What we learned
Reasoning models need to be told when *not* to reason. For a short structured answer, thinking mode costs the whole token budget and gives nothing back. And the bigger model is worth it for a low-resource language: Tamil quality was the deciding factor between the two Nemotron models, not speed.

### What's next for RustReader
- Caliper-measured real test set to report real-world accuracy, including how often Nemotron and OpenCV agree.
- Move the explainer to a Nebius Serverless Endpoint and the batch evaluation to Serverless Jobs.
- Bring the pipeline into the RustReader Flutter app (offline on-device scan + cloud deep scan with Nemotron).
- Shop stock lookup: scan → "available at 2 shops near you" with our industrial-supply partner.

## Feedback on Nebius Token Factory / NVIDIA models
1. `enable_thinking: false` via `chat_template_kwargs` is essential for short structured replies but is not obvious from the model cards. A note on the Token Factory model page ("this is a reasoning model; to get a direct answer send …") would have saved us the first hour.
2. Model ids are inconsistent between the two Nemotron 3.5 models we used: `Nemotron-3.5-Super-VL` uses a dot, `Nemotron-3_5-Lightning` uses an underscore. Easy to mistype.
3. The vision model sometimes returns `message.content` as a list of parts rather than a string; the OpenAI-compatible shape is otherwise exact. Worth documenting.
4. Tamil output from the 30B Lightning model drifts into other languages; the 120B Super VL is reliable. A language-quality note per model would help builders targeting Indian languages.
5. The OpenAI-compatible endpoint worked first time with plain `urllib` and no SDK — appreciated for a size-limited Lambda bundle.

## Built with
nvidia-nemotron, nebius-token-factory, opencv, python, numpy, aws-lambda, amazon-s3, aruco, html, javascript

## Try it out
- Live app: https://uyoot3fvlu4q7wimoneqsnrybu0akjof.lambda-url.ap-southeast-2.on.aws/
- Code (MIT): https://github.com/mylaimani/rustreader-nemotron
- Printable mat: live app → "Print the mat (A4)"
- Quick local test of the Nemotron calls: `python3 tools/test_nemotron.py samples/synthetic/M12x1.25x50_heavy.jpg`

## Video
(YouTube public link, 3 minutes or shorter)

## IRL event
(choose the Builders & Brews city if you attended one; otherwise leave blank)
