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
4. Deploy (CPU, models baked into the image; keep one warm instance during judging)
   ```bash
   gcloud run deploy calmline --source . --region asia-northeast3 --memory 8Gi --cpu 4 \
     --min-instances 1 --max-instances 3 --timeout 3600 --session-affinity \
     --set-env-vars CALMLINE_VERSION=v11 --set-secrets GEMINI_API_KEY=gemini-key:latest --allow-unauthenticated
   ```
   WebSockets work on Cloud Run; `--session-affinity` keeps a caller and agent on the same instance (rooms are in-memory).
5. Production STT: switch `Settings.spotter` / pass-2 to `calmline/audio/google_stt.py` (Speech-to-Text v2 streaming,
   `profanity_filter=False`). Sealed evidence → Cloud Storage bucket with CMEK + Firestore audit log (current code writes to `./evidence`).
