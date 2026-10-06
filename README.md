# CalmLine

**The industry trains the people being abused. CalmLine reduces the exposure.**

CalmLine sits between a caller and a public-service call agent (311, municipal complaint lines, utility
call centres). The caller's audio reaches the agent a couple of seconds late — like a broadcast delay.
Inside that window CalmLine

1. **transcribes** the call in streaming mode,
2. **mutes abusive words** before they reach the agent's ear and **turns shouting down** to conversational loudness,
3. hands the agent a **calm card**: the caller's request with every fact preserved (address, apartment, times,
   prior calls, who is affected, what they are asking for) and the abuse removed,
4. **escalates threats and self-harm** to a supervisor instead of softening them,
5. **seals the original** (audio + transcript) as evidence, and
6. leaves the agent **in control** — protection level, end-call request, audited reveal of the original.

Built for the AI Builder Cup 2026 (theme: *Sustainability & Social Impact*) on Google Cloud: Gemini / Gemma, Cloud Run,
Speech-to-Text, Firestore/Cloud Storage for sealed evidence.

## Why

| Evidence | Number |
|---|---|
| Verbal violence prevalence among public-sector workers | 19.4 – 63.5 %, ~3× the private sector |
| Government workforce survey: public incivility is a problem | 47 % |
| Call-centre interactions that are verbally aggressive | 15 – 20 % (≈10 calls a day) |
| 911 telecommunicators with PTSD symptoms | 18 – 24 % |
| Agencies whose answer is *more customer-service training for staff* | 73 % |

Sources are listed in `docs/sources.md`.

## How it works

```
caller audio ─┬─> streaming STT ─> committed words ─> abuse detector ─> mute intervals ─┐
              │                                     └─> utterance end ─> pipeline ─> calm card ─> agent UI
              ├─> prosody (loudness, pitch) ─> agitation gauge                          │
              └─> 2 s delay buffer ─────────────> [mute mask] ─> [look-ahead compressor] ┴─> agent's ear
```

**Text pipeline** (`calmline/pipeline.py`) — a 3-tier ladder so that *a classification failure is never a protection failure*:

| tier | when | output |
|---|---|---|
| 1 | intent recognised (26 intents covering **85.6 %** of NYC 311 volume, Jun–Aug 2026) | pre-written calm phrase for the issue + **verbatim** slots + cleaned detail clauses |
| 2 | intent unknown and an LLM is configured | Gemini / Gemma rewrite, **accepted only if it passes deterministic checks** (not abusive, no invented numbers, all slots kept) |
| 3 | otherwise | neutral header + the caller's own words with abuse removed |

Design rule: *templates carry the tone; slots carry the facts. Tone can be pre-written, facts can only be extracted.*

## Results

See [`results/RESULTS.md`](results/RESULTS.md) — every number is reproducible from a frozen version config
(`calmline/versions.py`, git tags `v0`…).

## Run locally

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
python scripts/download_models.py
.venv/bin/python -m uvicorn app.server:app --port 8765
# open http://localhost:8765/demo  -> press a sample call
```

Evaluate: `python -m eval.run_text v0 v4 v8 --splits dev,test` · Report: `python eval/report.py`

## Deploy (Cloud Run)

```bash
gcloud run deploy calmline --source . --region asia-northeast3 --memory 4Gi --cpu 4 --min-instances 1 \
  --set-env-vars CALMLINE_VERSION=v9 --set-secrets GEMINI_API_KEY=gemini-key:latest --allow-unauthenticated
```

## Honest limitations

* Test data is hand-written by the same person who wrote the lexicon → lexicon results are optimistic. External checks
  (Civil Comments, Gemma-written hard negatives, CREMA-D) are reported separately for that reason.
* Streaming experiments use TTS audio (no real anger prosody); acoustic aggression is evaluated on CREMA-D actors.
* No real 311 calls were used. The demo calls are fictional.
