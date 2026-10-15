# Deploying CalmLine on Google Cloud

1. Project + APIs
   ```bash
   gcloud config set project YOUR_PROJECT
   gcloud services enable run.googleapis.com aiplatform.googleapis.com speech.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
   ```
2. Gemini key (or use Vertex AI with `USE_VERTEX=1 GCP_PROJECT=...` and the service account's ADC)
   ```bash
   printf %s "$GEMINI_API_KEY" | gcloud secrets create gemini-key --data-file=-
   ```
3. S1 spike before anything else (does Gemini accept abusive input under BLOCK_NONE?)
   ```bash
   GEMINI_API_KEY=... .venv/bin/python scripts/gemini_s1_spike.py
   GEMINI_API_KEY=... .venv/bin/python -m eval.run_text v11 llm_all llm_raw --splits dev,test
   ```
4. Deploy (CPU, models baked into the image; keep one warm instance during judging). Gemini runs on Vertex AI with the
   service account's credentials, so no API key is stored anywhere:
   ```bash
   gcloud iam service-accounts create calmline-run
   for r in roles/aiplatform.user roles/speech.client roles/logging.logWriter; do
     gcloud projects add-iam-policy-binding YOUR_PROJECT --member=serviceAccount:calmline-run@YOUR_PROJECT.iam.gserviceaccount.com --role=$r
   done
   gcloud run deploy calmline --source . --region asia-northeast3 \
     --service-account calmline-run@YOUR_PROJECT.iam.gserviceaccount.com \
     --memory 4Gi --cpu 2 --min-instances 1 --max-instances 2 --concurrency 20 --timeout 3600 --session-affinity --cpu-boost \
     --set-env-vars USE_VERTEX=1,GCP_PROJECT=YOUR_PROJECT,GCP_LOCATION=global,CALMLINE_LIVE_LOCATION=us-central1,CALMLINE_ASR=google \
     --allow-unauthenticated
   ```
   WebSockets work on Cloud Run; `--session-affinity` keeps a caller and agent on the same instance (rooms are in-memory).
   * Vertex model availability: the Gemini 3.x text/TTS models are served from the `global` endpoint; the Live model
     (`gemini-live-2.5-flash-native-audio`) only from `us-central1`, hence the separate `CALMLINE_LIVE_LOCATION`.
   * Public-demo usage guard (per instance): `CALMLINE_MAX_CALL_S` (300), `CALMLINE_DAILY_CALL_MIN` (300),
     `CALMLINE_MAX_CONCURRENT` (4), `CALMLINE_DAILY_TYPED` (500). Set a billing budget alert as well.
5. Production STT: switch `Settings.spotter` / pass-2 to `calmline/audio/google_stt.py` (Speech-to-Text v2 streaming,
   `profanity_filter=False`). Sealed evidence → Cloud Storage bucket with CMEK + Firestore audit log (current code writes to `./evidence`).

## Google Cloud Speech-to-Text streaming (replaces the local Whisper passes)

Speech-to-Text does **not** accept API keys — it needs OAuth credentials (Application Default Credentials).

```bash
# 1. install the Google Cloud CLI (https://cloud.google.com/sdk/docs/install), then:
gcloud auth login
gcloud config set project YOUR_PROJECT            # a project with billing enabled
gcloud services enable speech.googleapis.com
gcloud auth application-default login             # creates ADC for the local server
gcloud auth application-default set-quota-project YOUR_PROJECT
```

Restart the server. `GET /api/health` should report `"asr": "google-stt-streaming"`.
`CALMLINE_ASR=whisper` forces the local path; `CALMLINE_ASR=google` forces Google.
Model: `CALMLINE_GOOGLE_STT_MODEL` (default `latest_short`; `phone_call` is an alternative for 8 kHz telephony audio).
On Cloud Run the service account's credentials are used automatically (grant it `roles/speech.client`).
