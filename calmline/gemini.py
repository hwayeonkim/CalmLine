"""Shared Gemini access: model fallback chains, retry on 429/503, and an on-disk cache.

The free tier rate-limits each model at ~15 requests/minute and preview models are sometimes overloaded,
so every call walks a fallback chain. Results are cached by (model-family, prompt) so a demo call that is
replayed costs nothing and plays back instantly."""
from __future__ import annotations
import hashlib
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache", "gemini")
TEXT_MODELS = os.getenv("CALMLINE_TEXT_MODELS", "gemini-3.5-flash-lite,gemini-flash-lite-latest,gemini-3.1-flash-lite").split(",")
TTS_MODELS = os.getenv("CALMLINE_TTS_MODELS", "gemini-3.8-flash-lite-tts,gemini-3.8-flash-tts,gemini-2.5-flash-preview-tts").split(",")


def load_env():
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p) and not os.getenv("GEMINI_API_KEY"):
        for line in open(p):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


_client = None


def client():
    global _client
    if _client is None:
        load_env()
        from google import genai
        if os.getenv("USE_VERTEX") == "1":
            _client = genai.Client(vertexai=True, project=os.environ["GCP_PROJECT"], location=os.getenv("GCP_LOCATION", "global"))
        else:
            _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


_live_client = None


def live_client():
    """Client for the Live API. On Vertex, Live models are regional (us-central1) while the Gemini 3.x text/TTS
    models are only served from the global endpoint, so Live gets its own client."""
    global _live_client
    if os.getenv("USE_VERTEX") != "1":
        return client()
    if _live_client is None:
        from google import genai
        _live_client = genai.Client(vertexai=True, project=os.environ["GCP_PROJECT"],
                                    location=os.getenv("CALMLINE_LIVE_LOCATION", "us-central1"))
    return _live_client


def available():
    load_env()
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("USE_VERTEX") == "1")


def _key(kind, payload):
    return hashlib.sha256(json.dumps([kind, payload], sort_keys=True).encode()).hexdigest()[:32]


def prompt_cache_on():
    """The prompt cache saves quota in offline evaluation. The demo server turns it off (CALMLINE_PROMPT_CACHE=0) so a
    live call always reaches Gemini; replayed sample calls use calmline.sample_cache instead."""
    return os.getenv("CALMLINE_PROMPT_CACHE", "1") == "1"


def cache_get(kind, payload, ext="json"):
    if not prompt_cache_on():
        return None
    p = os.path.join(CACHE, f"{kind}_{_key(kind, payload)}.{ext}")
    if os.path.exists(p):
        return open(p, "rb").read() if ext != "json" else json.load(open(p))
    return None


def cache_put(kind, payload, value, ext="json"):
    if not prompt_cache_on():
        return
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{kind}_{_key(kind, payload)}.{ext}")
    if ext == "json":
        json.dump(value, open(p, "w"))
    else:
        open(p, "wb").write(value)


def _retryable(e):
    s = str(e)
    return any(c in s for c in ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"))


_dead_until: dict = {}   # circuit breaker: model -> time before which it is skipped


def _trip(model, err):
    s = str(err)
    if "PerDay" in s:
        _dead_until[model] = time.time() + 3600      # daily quota gone: skip for an hour
    elif "429" in s or "503" in s or "UNAVAILABLE" in s:
        _dead_until[model] = time.time() + 30


def generate_text(prompt: str, config=None, models=None, use_cache=True, retry=True):
    """Returns (text, model_used). Walks the fallback chain (skipping tripped models); one backoff round if all are busy."""
    payload = {"prompt": prompt}
    if use_cache:
        hit = cache_get("text", payload)
        if hit:
            return hit["text"], hit["model"] + " (cached)"
    last = None
    for attempt in range(2 if retry else 1):
        for m in (models or TEXT_MODELS):
            if _dead_until.get(m, 0) > time.time():
                continue
            try:
                r = client().models.generate_content(model=m, contents=prompt, config=config)
                text = (r.text or "").strip()
                if use_cache and text:
                    cache_put("text", payload, {"text": text, "model": m})
                return text, m
            except Exception as e:  # noqa: BLE001
                last = e
                _trip(m, e)
                if not _retryable(e) and "404" not in str(e):
                    raise
        if retry:
            time.sleep(4 + 4 * attempt)
    raise last or RuntimeError("all Gemini models unavailable")
