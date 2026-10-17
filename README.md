# CalmLine

Prototype for the AI Builder Cup 2026 (Sustainability & Social Impact). All rights reserved; shared for evaluation.

CalmLine sits between a caller and a public-service call agent. The agent does not hear the caller's raw audio:
each part of what the caller says is transcribed, checked for abuse, and relayed to the agent as a calm spoken
message that keeps the facts, together with a service request filled in from the call. Threats are not softened;
they raise an alert.

The demo uses fictional calls about New York City 311 issues (heating, parking, potholes, noise, sanitation).

## How a call is processed

```
caller audio ─> Google Cloud Speech-to-Text (streaming) ─> clause-level parts
   part ─> fast detector (rules + RoBERTa) ─┬─ clean            ─> relayed as is
                                            ├─ profanity only   ─> cleaned offline
                                            ├─ abuse only       ─> withheld
                                            ├─ insult / sarcasm ─> Gemini (≤ 2.5 s), output verified
                                            └─ threat           ─> safety alert
   relayed text ─> Gemini Live voice ─> agent        Gemini structured output ─> service request
```

* **Parts, not sentences.** Speech is cut at clause boundaries (clause punctuation or a short pause) so the
  agent hears each part shortly after the caller says it.
* **Understanding.** One structured Gemini call per part returns the complaint type (29 types based on NYC 311
  categories), the facts (address, unit, timing, prior contact) and a calm message. A fact is kept only if it
  appears in the caller's words.
* **Verification.** A Gemini message is spoken only if no abuse remains and it contains no number the caller did not
  say; otherwise a template built from the caller's own cleaned words is used. If Gemini is late, the part goes out
  on the offline path and the late answer only updates the service request.
* **Sample calls** replay stored results by default to save API cost in the public demo (labeled "Cached replay";
  switch it off to process the sample live). Microphone and typed input are always processed live.

## Google Cloud components

Cloud Run (FastAPI + WebSockets), Speech-to-Text streaming, Vertex AI Gemini (understanding and calm rewrite),
Vertex AI Gemini Live (calm voice), Cloud Storage (sample replay cache), Secret Manager, GitHub Actions deployment.
Configuration: [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Repository layout

| Path | Contents |
|---|---|
| `calmline/` | detection, understanding, rewriting, verification, session handling, speech recognition |
| `app/` | server and demo pages |
| `calmline/versions.py` | frozen configurations of every evaluated version (git tags `v0` to `v20`) |
| `eval/`, `results/` | evaluation code and results ([`results/RESULTS.md`](results/RESULTS.md)) |
| `data/gold`, `data/synth` | hand-written and synthetic evaluation sets |

## Run locally

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_models.py
.venv/bin/python -m uvicorn app.server:app --port 8765
```

Open `http://localhost:8765/demo`. Gemini needs `GEMINI_API_KEY` in `.env` or Vertex AI (`USE_VERTEX=1`,
`GCP_PROJECT`). Speech-to-Text needs Application Default Credentials; without them the server falls back to a local
Whisper recognizer.

Text evaluation: `.venv/bin/python -m eval.run_text v0 v10 v20 --splits dev,test`

## Limitations

* No real 311 calls were used. Demo calls are fictional and voiced by Gemini; evaluation sets are hand-written or
  synthetic, so results are indicative rather than field performance.
* Versions v11 and later were developed after reading test-set errors, so their test results are optimistic.
* Latency figures come from local runs; deployed microphone latency has not been benchmarked.
* Supervisor handoff, two-way telephony and protected evidence storage are not implemented.
