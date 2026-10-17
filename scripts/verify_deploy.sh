#!/usr/bin/env bash
# Check the live Cloud Run deployment against docs/DEPLOY.md. Read-only; prints one line per check.
# Exit code 0 = everything matches. Needs gcloud (logged in), curl and python3.
set -uo pipefail

PROJECT=${PROJECT:-project-a86fde9c-208a-4edc-a0e}
REGION=${REGION:-asia-northeast3}
SERVICE=${SERVICE:-calmline}
URL=${URL:-https://calmline-930584488838.asia-northeast3.run.app}
BUCKET=${BUCKET:-calmline-sample-cache-a86fde9c}
SECRET=${SECRET:-calmline-admin-token}
RUN_SA="calmline-run@${PROJECT}.iam.gserviceaccount.com"

fail=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fail=1; }

svc=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format=json) || { echo "cannot read service"; exit 2; }

echo "== service configuration"
SVC_JSON="$svc" python3 - "$RUN_SA" "$PROJECT" "$BUCKET" <<'EOF' || fail=1
import json, os, sys
d = json.loads(os.environ["SVC_JSON"]); run_sa, project, bucket = sys.argv[1:4]
t = d["spec"]["template"]; spec = t["spec"]; c = spec["containers"][0]; a = t["metadata"].get("annotations", {})
env = {e["name"]: e for e in c.get("env", [])}
want_env = {"USE_VERTEX": "1", "GCP_PROJECT": project, "GCP_LOCATION": "global",
            "CALMLINE_LIVE_LOCATION": "us-central1", "CALMLINE_ASR": "google",
            "CALMLINE_GOOGLE_STT_MODEL_MIC": "latest_long", "CALMLINE_CACHE_BUCKET": bucket}
checks = [
    ("service account = " + run_sa, spec.get("serviceAccountName") == run_sa),
    ("cpu 2 / memory 4Gi", c["resources"]["limits"] == {"cpu": "2", "memory": "4Gi"}),
    ("min 1 / max 2 instances", (a.get("autoscaling.knative.dev/minScale"), a.get("autoscaling.knative.dev/maxScale")) == ("1", "2")),
    ("concurrency 20 / timeout 3600", (spec.get("containerConcurrency"), spec.get("timeoutSeconds")) == (20, 3600)),
    ("session affinity on", a.get("run.googleapis.com/sessionAffinity") == "true"),
    ("startup cpu boost on", a.get("run.googleapis.com/startup-cpu-boost") == "true"),
    ("traffic: 100% to latest revision", d["spec"].get("traffic") == [{"latestRevision": True, "percent": 100}]),
]
for name, want in want_env.items():
    have = env.get(name, {}).get("value")
    checks.append((f"env {name} = {want}", have == want))
checks.append(("env CALMLINE_GOOGLE_STT_MODEL unset (samples use latest_short)", "CALMLINE_GOOGLE_STT_MODEL" not in env))
adm = env.get("CALMLINE_ADMIN_TOKEN", {}).get("valueFrom", {}).get("secretKeyRef", {})
checks.append(("secret CALMLINE_ADMIN_TOKEN -> calmline-admin-token:latest (optional)", adm.get("name") == "calmline-admin-token"))
bad = 0
for name, good in checks:
    print(("  ok    " if good else "  FAIL  ") + name); bad |= not good
print(f"  info  serving revision {d['status'].get('latestReadyRevisionName')}")
sys.exit(1 if bad else 0)
EOF

echo "== access and IAM"
gcloud run services get-iam-policy "$SERVICE" --project "$PROJECT" --region "$REGION" --format=json \
  | grep -q '"allUsers"' && ok "public: allUsers has run.invoker" || bad "public access (allUsers run.invoker) missing"
gcloud storage buckets get-iam-policy "gs://$BUCKET" --format=json 2>/dev/null \
  | python3 -c "import json,sys; p=json.load(sys.stdin); sys.exit(0 if any(b['role']=='roles/storage.objectAdmin' and 'serviceAccount:$RUN_SA' in b['members'] for b in p.get('bindings',[])) else 1)" \
  && ok "bucket gs://$BUCKET: runtime SA has storage.objectAdmin" || bad "bucket gs://$BUCKET missing or runtime SA lacks storage.objectAdmin"
gcloud secrets get-iam-policy "$SECRET" --project "$PROJECT" --format=json 2>/dev/null \
  | python3 -c "import json,sys; p=json.load(sys.stdin); sys.exit(0 if any(b['role']=='roles/secretmanager.secretAccessor' and 'serviceAccount:$RUN_SA' in b['members'] for b in p.get('bindings',[])) else 1)" \
  && ok "secret $SECRET: runtime SA has secretAccessor" || bad "secret $SECRET missing or runtime SA lacks secretAccessor"
gcloud artifacts repositories describe cloud-run-source-deploy --project "$PROJECT" --location "$REGION" --format=json 2>/dev/null \
  | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('cleanupPolicies') and not d.get('cleanupPolicyDryRun') else 1)" \
  && ok "artifact registry cleanup policy enforced" || bad "artifact registry cleanup policy missing or in dry-run"

echo "== live endpoints"
health=$(curl -fsS --max-time 20 "$URL/api/health") && ok "GET /api/health" || bad "GET /api/health"
HEALTH="${health:-}" python3 - <<'EOF' || fail=1
import json, os
h = json.loads(os.environ["HEALTH"] or "{}")
m = h.get("stt_model", {})
for name, good in [("health ok=true, asr=google-stt-streaming", h.get("ok") is True and h.get("asr") == "google-stt-streaming"),
                   ("stt model: sample=latest_short, microphone=latest_long", m == {"sample": "latest_short", "microphone": "latest_long"})]:
    print(("  ok    " if good else "  FAIL  ") + name + ("" if good else f"  (got {h})"))
    if not good: raise SystemExit(1)
EOF
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$URL/demo"); [ "$code" = 200 ] && ok "GET /demo 200" || bad "GET /demo -> $code"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$URL/api/admin/cache"); [ "$code" = 401 ] && ok "admin API needs the token (401)" || bad "admin API without token -> $code (want 401)"

echo
[ $fail = 0 ] && echo "All checks passed." || echo "Some checks FAILED: see docs/DEPLOY.md section 4 for the repair command."
exit $fail
