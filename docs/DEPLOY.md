# Deploying CalmLine on Google Cloud

This is the single source of truth for the production deployment. Anything set by hand (env vars, secrets, IAM,
traffic) is listed here; `scripts/verify_deploy.sh` checks the live service against it. **Run that script after every
deploy or config change.**

Production: project `project-a86fde9c-208a-4edc-a0e`, region `asia-northeast3`, Cloud Run service `calmline`,
public URL `https://calmline-930584488838.asia-northeast3.run.app` (stable across deploys; only revision names change).

## 1. Architecture

```
browser (demo console)  ──WebSocket──>  Cloud Run "calmline" (Seoul, 2 vCPU / 4 GiB, 1–2 instances)
  /ws/sample  sample calls                ├─ Speech-to-Text v1 streaming      (per-mode model, see §3)
  /ws/caller  microphone / typed text     ├─ Vertex AI Gemini 3.5 Flash-Lite  (global)       understanding + calm rewrite
  /ws/agent   agent view                  ├─ Vertex AI Gemini Live 2.5        (us-central1)  verbatim calm voice
                                          ├─ Vertex AI Gemini TTS             (global)       fallback voice, fillers
                                          ├─ open models baked into the image (toxicity, intent, Whisper fallback)
                                          ├─ Cloud Storage bucket             sample-call replay cache
                                          └─ Secret Manager                   admin page token
```

* All Google AI calls use the runtime service account (`calmline-run`) through Vertex AI / ADC. **No API keys.**
* Call rooms are in memory: `--session-affinity` keeps a caller and its agent view on one instance.
* GitHub Actions rebuilds the image on every push to `main` (§5). Everything else in this file is configuration that
  lives on the service or in IAM and is **not** in the image.

## 2. Resources that must exist

| Resource | Name | Notes |
|---|---|---|
| APIs | run, cloudbuild, artifactregistry, aiplatform, speech, secretmanager, storage, iamcredentials, sts, billingbudgets | `gcloud services enable …` |
| Runtime service account | `calmline-run@<project>.iam.gserviceaccount.com` | project roles: `aiplatform.user`, `speech.client`, `logging.logWriter` |
| Deployer service account | `calmline-deployer@<project>.iam.gserviceaccount.com` | `run.sourceDeveloper`, `serviceusage.serviceUsageConsumer`, `logging.viewer`; `iam.serviceAccountUser` on `calmline-run` **and** on the Cloud Build SA `<number>-compute@developer.gserviceaccount.com` |
| Cloud Build SA | `<number>-compute@developer.gserviceaccount.com` | `roles/run.builder` (builds `--source` deploys) |
| Workload Identity Federation | pool `github`, provider `calmline-repo` | condition `assertion.repository=='hwayeonkim/CalmLine' && assertion.ref=='refs/heads/main'`; `workloadIdentityUser` on the deployer for `attribute.repository/hwayeonkim/CalmLine` |
| Artifact Registry | `cloud-run-source-deploy` (asia-northeast3) | cleanup policy: keep 3 most recent `calmline` images, delete older (`keep-3-most-recent`, `delete-older`) |
| Cache bucket | `gs://calmline-sample-cache-a86fde9c` | uniform access; `calmline-run` has `roles/storage.objectAdmin` on the bucket |
| Secret | `calmline-admin-token` | `calmline-run` has `roles/secretmanager.secretAccessor` on the secret |
| Budget | `calmline-budget` on the billing account | ₩400,000/month, alerts at 20/50/80/100 %, credits **excluded** (otherwise the free-trial spend shows as ₩0) |

## 3. Per-mode decisions

| Mode | Path | Speech-to-Text model | Why |
|---|---|---|---|
| Sample calls | `/ws/sample` | `latest_short` (`CALMLINE_GOOGLE_STT_MODEL`, default) | Replay cache entries were recorded with it; the cache key does not include the model. |
| Microphone calls | `/ws/caller` | **`latest_long`** (`CALMLINE_GOOGLE_STT_MODEL_MIC`) | A/B tests on the production recognizer: `latest_short` dropped whole clauses of live speech ("No heat at …", "We have had no heat for three days …") even when its interim had them; `latest_long` kept them (9/9 vs 6/9 facts; 20/20 vs 19/20 on five new recordings) at the same finalization delay. It makes some word-level errors `latest_short` does not ("no heed"), so it is not the default. |
| Typed text | `/ws/caller` (text) | none | No STT. |

Shared by all modes:

* Sentence parts: cut after a 0.45 s pause (`CALMLINE_GOOGLE_PAUSE_S`) or at a clause boundary (v18 clause chunking,
  `CALMLINE_GOOGLE_CLAUSE_PAUSE_S` 0.25 s). The stream is half-closed at each cut so Google finalizes immediately.
* Calm voice: Gemini Live `gemini-live-2.5-flash-native-audio` in `us-central1` (`CALMLINE_LIVE_LOCATION`), one
  fresh session per sentence with a plain "Read aloud" prompt; Gemini TTS is the fallback, then the browser voice.
* Text: Gemini 3.x Flash-Lite on the `global` endpoint (`GCP_LOCATION=global`). Live models are not served there.
* Sample calls: replayed from the cache bucket when "Cached replay" is ON (§6); pause / resume / stop supported.
* Public-demo usage guard (per instance, defaults in code): 5 min per call (`CALMLINE_MAX_CALL_S`), 300 call-minutes
  per UTC day (`CALMLINE_DAILY_CALL_MIN`), 4 concurrent calls (`CALMLINE_MAX_CONCURRENT`), 500 typed messages
  (`CALMLINE_DAILY_TYPED`). Cached sample replays do not count.

## 4. Service configuration

Set once; `gcloud run deploy --source` from CI keeps all of it.

| Setting | Value |
|---|---|
| Service account | `calmline-run@<project>.iam.gserviceaccount.com` |
| CPU / memory | 2 vCPU / 4 GiB, startup CPU boost |
| Instances | min 1 (no cold start for judges), max 2 (also the spend ceiling) |
| Concurrency / timeout | 20 / 3600 s |
| Session affinity | on |
| Ingress / access | all; `allUsers` has `roles/run.invoker` (public demo) |
| Traffic | **100 % to the latest revision** (`--to-latest`) |

Environment variables and secrets:

| Name | Value | Required |
|---|---|---|
| `USE_VERTEX` | `1` | yes |
| `GCP_PROJECT` | `project-a86fde9c-208a-4edc-a0e` | yes |
| `GCP_LOCATION` | `global` | yes |
| `CALMLINE_LIVE_LOCATION` | `us-central1` | yes |
| `CALMLINE_ASR` | `google` | yes |
| `CALMLINE_GOOGLE_STT_MODEL_MIC` | `latest_long` | yes |
| `CALMLINE_CACHE_BUCKET` | `calmline-sample-cache-a86fde9c` | yes (otherwise the cache is lost on every revision) |
| `CALMLINE_ADMIN_TOKEN` | secret `calmline-admin-token:latest` | optional (without it `/admin` is 404) |
| `CALMLINE_GOOGLE_STT_MODEL` | unset (= `latest_short`) | leave unset |

Apply or repair the whole set in one command (idempotent; creates a new revision):

```bash
gcloud run services update calmline --region asia-northeast3 \
  --service-account calmline-run@project-a86fde9c-208a-4edc-a0e.iam.gserviceaccount.com \
  --cpu 2 --memory 4Gi --min-instances 1 --max-instances 2 --concurrency 20 --timeout 3600 --session-affinity --cpu-boost \
  --update-env-vars USE_VERTEX=1,GCP_PROJECT=project-a86fde9c-208a-4edc-a0e,GCP_LOCATION=global,CALMLINE_LIVE_LOCATION=us-central1,CALMLINE_ASR=google,CALMLINE_GOOGLE_STT_MODEL_MIC=latest_long,CALMLINE_CACHE_BUCKET=calmline-sample-cache-a86fde9c \
  --update-secrets CALMLINE_ADMIN_TOKEN=calmline-admin-token:latest
gcloud run services update-traffic calmline --region asia-northeast3 --to-latest
gcloud run services add-iam-policy-binding calmline --region asia-northeast3 --member=allUsers --role=roles/run.invoker
```

Use `--update-env-vars` / `--update-secrets` (merge), never `--set-env-vars` (replaces the whole list and silently
drops everything not named).

## 5. How deploys happen

* **Code**: merge to `main` → `.github/workflows/deploy.yml` builds from source, deploys, routes traffic to the latest
  revision and calls `/api/health`. Docs-only pushes (`**.md`, `docs/`, `results/`, `eval/`, `experiments/`) are
  skipped; run it by hand from the Actions tab (workflow_dispatch on `main`). Auth is GitHub OIDC → Workload Identity
  Federation; there are no keys. The deploy replaces only the image.
* **Configuration**: `gcloud run services update …` (§4). Creates a revision with the same image.
* **Preview without traffic**: `gcloud run deploy calmline --source . --no-traffic --tag preview` gives
  `https://preview---…run.app`. This pins traffic to the revision serving at that moment, so afterwards run
  `gcloud run services update-traffic calmline --region asia-northeast3 --to-latest --remove-tags preview`
  (the CI workflow also does `--to-latest` on its next run).

After any of these: `scripts/verify_deploy.sh` (exit code 0 = matches this document).

## 6. Sample-call cache and admin page

Built-in sample calls are replayed from a cache by default: the first play of a sample (with "Cached replay" ON) runs
live and stores its Speech-to-Text parts, cards and calm voice; later plays reuse them without calling Speech-to-Text,
Gemini or Gemini Live, and do not count toward `CALMLINE_DAILY_CALL_MIN`. Switching "Cached replay" OFF in the demo
runs every part live and stores nothing. Microphone and typed input are never cached. The offline-evaluation prompt
cache (`data/cache/gemini`) is off in the server (`CALMLINE_PROMPT_CACHE=0`), so live mode always reaches Gemini.

Entries are keyed by pipeline version, recognizer (`google-stt-streaming` / `whisper-local`), sample audio hash and a
format epoch (`calmline/sample_cache.py: EPOCH`), so a changed sample or recognizer never replays stale results. A
change to prompts, cut rules or the **sample-call STT model** under the same version does not change the key: clear
the cache after such a change.

* Storage: the bucket in §2 (`CALMLINE_CACHE_BUCKET`). Without it the cache lives in the instance's memory, is lost on
  every restart or new revision, and is not shared between instances.
* Admin page `/admin` (cache contents per sample, clear one sample or everything) exists only when
  `CALMLINE_ADMIN_TOKEN` is set; without it `/admin` and `/api/admin/*` return 404. Read the token with
  `gcloud secrets versions access latest --secret calmline-admin-token`; turn the page off with
  `gcloud run services update calmline --region asia-northeast3 --remove-secrets CALMLINE_ADMIN_TOKEN`.
* After a clear (or a change listed above), play each sample once with "Cached replay" ON, start to finish without
  pause or stop, and check `/admin` shows the transcript as stored.

## 7. Rollback

* Microphone model: `gcloud run services update calmline --region asia-northeast3 --remove-env-vars CALMLINE_GOOGLE_STT_MODEL_MIC`
  (microphone calls go back to `latest_short`).
* Code: `gcloud run services update-traffic calmline --region asia-northeast3 --to-revisions <previous-revision>=100`,
  then fix forward on `main` (the next CI deploy routes traffic back to the latest revision). Only the 3 most recent
  images are kept.

## 8. First-time setup from scratch (another project)

```bash
P=YOUR_PROJECT; R=asia-northeast3; N=$(gcloud projects describe $P --format='value(projectNumber)')
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com \
  speech.googleapis.com secretmanager.googleapis.com storage.googleapis.com iamcredentials.googleapis.com sts.googleapis.com
gcloud iam service-accounts create calmline-run
for r in roles/aiplatform.user roles/speech.client roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding $P --member=serviceAccount:calmline-run@$P.iam.gserviceaccount.com --role=$r; done
gcloud projects add-iam-policy-binding $P --member=serviceAccount:$N-compute@developer.gserviceaccount.com --role=roles/run.builder
gcloud run deploy calmline --source . --region $R --service-account calmline-run@$P.iam.gserviceaccount.com --allow-unauthenticated
# then: bucket + secret (§2, §6), the full configuration command (§4), CI identity (§2) and scripts/verify_deploy.sh
```

Vertex model availability changes over time: the Gemini 3.x text/TTS models were on the `global` endpoint and the Live
model only in `us-central1` when this was set up. Re-check with a smoke call before moving regions.

## Running locally with Google Speech-to-Text

Speech-to-Text does not accept API keys; it needs Application Default Credentials.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT
```

`GET /api/health` should report `"asr": "google-stt-streaming"` and the model per mode. `CALMLINE_ASR=whisper` forces
the local Whisper path.
