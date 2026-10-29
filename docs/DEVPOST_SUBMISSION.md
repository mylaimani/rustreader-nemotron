# Devpost submission — copy/paste fields

## Project name
RustReader Deep Scan

## Elevator pitch (short tagline)
Point your phone at a rusted bolt nobody can read. An OpenCV 5 agent on AWS measures its thread size, pitch and length, judges the rust, and tells you what to buy.

## Featured path
Agentic Vision with OpenCV 5

## About the project

### Inspiration
Old fasteners rust, and rust hides everything: head markings, thread pitch, sometimes even the size. A relative of mine who works in fastener manufacturing could not identify a battery bolt worth about ₹30; the "closest fit" he used instead risked a ₹5,000 repair. Mechanics here travel 15 km to hardware shops carrying the old bolt to match it. Existing fastener-ID apps are trained on clean catalogue photos and fail on exactly these rusted parts.

### What it does
Place the bolt on a printed A4 mat (four ArUco markers) and take one photo. RustReader Deep Scan:
- straightens the photo and converts pixels to millimetres,
- measures shank diameter, length and head type (hex, socket cap, countersunk, flanged),
- reads the thread pitch with an FFT of the thread edge, even under rust,
- scores rust severity and shows a "Rust Mode" enhanced view,
- matches ISO metric coarse/fine and UNC tables, e.g. **M12x1.25 x 50 hex bolt, spanner 18 mm**,
- recommends replace or reuse (advisory) and links to nearby hardware shops or an online supplier.

The agentic part: **what OpenCV sees decides what happens next**. Missing markers, blur, a tilted or folded mat, or two bolts in view lead to a specific request for a new photo. Heavy rust switches thread reading to a CLAHE + Sobel sub-pixel edge tool, and a weak FFT peak triggers a retry with the other tool. If the threads are still unreadable it asks for a closer, side-lit photo and fuses both photos. If two sizes still fit, it asks the human to choose instead of guessing. Every step is logged in a decision trace shown in the app.

### How we built it
- **OpenCV 5.0** (`opencv-python-headless 5.0.0.93`): ArUco detection with sub-pixel corners, RANSAC homography, perspective rectification to 12 px/mm, Laplacian sharpness, a local-background + HSV segmentation, PCA alignment, width-profile measurement, CLAHE/bilateral/Sobel edge refinement, HSV rust analysis.
- **Agent**: a perception → decision → action loop in Python; each OpenCV tool returns observations; a transparent policy decides the next tool call or user action; the trace is stored per session.
- **AWS**: AWS Lambda (Python 3.12, **arm64 Graviton**) behind a Lambda Function URL serves both the web app and the API; Amazon S3 stores sessions, decision traces and scans (EXIF stripped, 30-day expiry); Amazon CloudWatch collects Embedded Metric Format metrics (latency, agent outcomes) on a dashboard. One idempotent `deploy.sh` builds the OpenCV 5 arm64 bundle and deploys everything.
- **Evaluation**: a synthetic generator renders bolts with exact known geometry, rust and perspective; plus real rusted fasteners measured with a caliper.

### Challenges we ran into
- Rust destroys thread edges. A single binary outline was not enough for heavy rust, so the agent chooses between two edge tools and checks FFT peak confidence.
- Countersunk screws: the cone hides where the head ends, and their nominal length includes the head. The head shape is now classified from the side profile.
- Uneven workshop lighting: a single threshold failed, so we model the paper brightness locally with a large median filter.
- Real shadows: in our field photos the bolt's shadow was segmented as metal. We now require a pixel to be darker than about half the local paper brightness, then grow the outline back to the true half-way edge within 0.8 mm.
- Deploying OpenCV 5 to Lambda without Docker: we bundle the manylinux aarch64 wheels (146 MB unzipped) via S3.

### Accomplishments that we're proud of
- Correct thread spec on 8/8 synthetic bolts with exact ground truth (fine thread under heavy rust, inch UNC, partially threaded, countersunk, socket head), pitch error ≤ 0.01 mm, and 3/3 bad photos correctly sent back.
- A field test with real fasteners from a local mechanic found four failure modes synthetic data missed (cast shadows read as metal, grey-printed markers misjudged as blur, shiny zinc shanks splitting one bolt into two, rippled paper). Three are fixed and the fourth is correctly caught by the agent. The same flange bolt photographed separately is identified consistently as M8 x 60 (pitch 1.29 / 1.31 mm). See report section 7.2.
- Full round trip on Lambda (Graviton) under 1 second.

### What we learned
Measurement beats classification for this problem: with a calibrated mat, size is geometry, not guesswork, and needs no training data. And an agent that knows when to ask for a better photo is more useful than one that always answers.

### What's next for RustReader
- Caliper-measured real test set to report real-world accuracy.
- Bring this pipeline into the RustReader Flutter app (offline on-device scan + cloud deep scan).
- Nuts, washers, pan/button heads; thread-pitch from a close-up without the mat.
- Shop stock lookup: scan → "available at 2 shops near you" with our industrial-supply partner.

## Built with
opencv, python, numpy, aws-lambda, aws-graviton, amazon-s3, amazon-cloudwatch, aruco, html, javascript

## Try it out
- Live app: https://5t2v7gt4kwkghxncjnzdd6g5zi0kldaf.lambda-url.ap-southeast-2.on.aws/
- Code: https://github.com/mylaimani/rustreader-deep-scan
- Printable mat: (live app → "Print the mat (A4)")

## Video
(YouTube unlisted link)

## Files to attach (if the form asks)
- docs/TECHNICAL_REPORT.md (export to PDF)
- docs/architecture.png, docs/agent_workflow.png
