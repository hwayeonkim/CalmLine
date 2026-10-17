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
   * Continuous deployment: `.github/workflows/deploy.yml` redeploys on every push to `main` through GitHub OIDC ->
     Workload Identity Federation (pool `github`, provider `calmline-repo`, restricted to this repo's `main`) as the
     `calmline-deployer` service account (`roles/run.sourceDeveloper`, `roles/serviceusage.serviceUsageConsumer`,
     `roles/logging.viewer`, `roles/iam.serviceAccountUser` on `calmline-run`). It only replaces the image; env vars,
     scaling and public access stay as configured. Run it by hand from the Actions tab (workflow_dispatch on `main`).
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

## Sample-call cache and admin page

Built-in sample calls are replayed from a cache by default: the first play of a sample (with "Cached replay" ON) runs
live and stores its Speech-to-Text parts, cards and calm voice; later plays reuse them without calling Speech-to-Text,
Gemini or Gemini Live, and do not count toward `CALMLINE_DAILY_CALL_MIN`. Switching "Cached replay" OFF in the demo
runs every part live and stores nothing. Microphone and typed input are never cached. The offline-evaluation prompt
cache (`data/cache/gemini`) is off in the server (`CALMLINE_PROMPT_CACHE=0`), so live mode always reaches Gemini.

Entries are keyed by pipeline version, recognizer (`google-stt-streaming` / `whisper-local`), sample audio hash and a
format epoch (`calmline/sample_cache.py: EPOCH`), so a changed sample or recognizer never replays stale results. A
change to prompts or cut rules under the same version does: clear the cache after such a change.

* Storage: `data/sample_cache/` by default. On Cloud Run that folder lives in the instance's memory, is lost on every
  restart or new revision, and is not shared between instances. For a persistent shared cache set
  `CALMLINE_CACHE_BUCKET=<bucket>` and grant the runtime service account `roles/storage.objectAdmin` on that bucket:
  ```bash
  gcloud storage buckets create gs://YOUR_BUCKET --location asia-northeast3 --uniform-bucket-level-access
  gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET --member=serviceAccount:calmline-run@YOUR_PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin
  gcloud run services update calmline --region asia-northeast3 --update-env-vars CALMLINE_CACHE_BUCKET=YOUR_BUCKET
  ```
* Admin page `/admin` (cache contents per sample, clear one sample or everything) exists only when
  `CALMLINE_ADMIN_TOKEN` is set; without it `/admin` and `/api/admin/*` return 404, so the public deployment has no
  clear function. Keep the token in Secret Manager (`--update-secrets CALMLINE_ADMIN_TOKEN=calmline-admin-token:latest`).
* After a deploy or a clear, play each sample once with "Cached replay" ON so judges get the stored replay.
