# Demo video (max 5 minutes) — shot list

Record screen + phone. Upload to YouTube as **Unlisted**.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:30 | You, holding a rusted bolt | "I'm Mani, a software developer. This ₹30 bolt nearly cost a ₹5,000 repair because nobody could read it. RustReader identifies rusted bolts from one photo." |
| 0:30–1:00 | Printed mat, phone | "Print this A4 mat. Four ArUco markers give OpenCV the scale and perspective. Put one bolt on it and take a photo." |
| 1:00–2:15 | Phone: open the Function URL, scan a real rusty bolt | Result: part name, pitch, length, rust %, "replace recommended", find-nearby button. Compare with your caliper reading. |
| 2:15–3:00 | Show the agent asking: two bolts on mat, then blurry photo, then fixed photo | "This is the agentic part: what OpenCV sees changes what happens next: new photo, a different edge tool for heavy rust, or asking me to choose." |
| 3:00–3:40 | Decision trace on screen; Rust map / Rust Mode tabs | Walk through 3–4 trace steps; point at "rust is heavy, so started with the enhanced edge tool". |
| 3:40–4:20 | docs/architecture.png, AWS console: Lambda (arm64), S3 bucket, CloudWatch dashboard | "OpenCV 5 runs on AWS Lambda on Graviton; sessions and traces go to S3; metrics to CloudWatch." |
| 4:20–5:00 | Evaluation table + limitations | Accuracy on real bolts, one failure case, "advisory only, human decides when unsure". Close with the Industrial 360 / nearby shop link. |
