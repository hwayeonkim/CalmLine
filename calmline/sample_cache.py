"""Replay cache for the built-in sample calls.

A sample call is the same audio every time, so its speech-to-text parts, its cards (Gemini understanding + checks) and
the calm voice for each part are stored the first time it is played and replayed afterwards without calling Speech-to-
Text, Gemini or Gemini Live. Microphone and typed input are never cached.

Layout (one namespace per sample, pipeline version, ASR mode and audio file, so a stale entry is never mixed in):
    <EPOCH>/<version>-<asr>/<sample>-<wav sha8>/stt.json          parts as the recognizer emitted them, with call time
    <EPOCH>/<version>-<asr>/<sample>-<wav sha8>/cards/<key>.json   card (+ late card update) per (part index, text)
    <EPOCH>/<version>-<asr>/<sample>-<wav sha8>/voice/<key>.pcm    calm voice, PCM16 @16 kHz, per spoken text

Storage: a local folder (data/sample_cache) by default, or a Cloud Storage bucket when CALMLINE_CACHE_BUCKET is set.
On Cloud Run the local folder lives in the instance's memory and is lost on restart; a bucket is shared and persistent.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time

import numpy as np

EPOCH = "e1"     # bump when the cached formats or the meaning of a cached result change
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_cache")
_lock = threading.Lock()


def _h(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]


class _Local:
    name = "local"

    def __init__(self, root=ROOT):
        self.root = root

    def get(self, key):
        p = os.path.join(self.root, key)
        try:
            with open(p, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def put(self, key, data: bytes):
        p = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p))      # atomic: a reader never sees half a file
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, p)

    def list(self, prefix=""):
        base = os.path.join(self.root, prefix)
        out = []
        for d, _, files in os.walk(base):
            for fn in files:
                p = os.path.join(d, fn)
                out.append((os.path.relpath(p, self.root), os.path.getsize(p), os.path.getmtime(p)))
        return out

    def delete_prefix(self, prefix=""):
        n = 0
        for key, _, _ in self.list(prefix):
            try:
                os.remove(os.path.join(self.root, key)); n += 1
            except FileNotFoundError:
                pass
        return n


class _GCS:
    name = "gcs"

    def __init__(self, bucket, prefix="sample_cache/"):
        from google.cloud import storage
        self.b = storage.Client().bucket(bucket)
        self.prefix = prefix

    def get(self, key):
        blob = self.b.blob(self.prefix + key)
        try:
            return blob.download_as_bytes()
        except Exception:  # noqa: BLE001  (NotFound or transient: treat as a miss)
            return None

    def put(self, key, data: bytes):
        self.b.blob(self.prefix + key).upload_from_string(data)

    def list(self, prefix=""):
        return [(b.name[len(self.prefix):], b.size or 0, b.updated.timestamp() if b.updated else 0)
                for b in self.b.list_blobs(prefix=self.prefix + prefix)]

    def delete_prefix(self, prefix=""):
        blobs = list(self.b.list_blobs(prefix=self.prefix + prefix))
        for b in blobs:
            b.delete()
        return len(blobs)


_store = None


def store():
    global _store
    with _lock:
        if _store is None:
            bucket = os.getenv("CALMLINE_CACHE_BUCKET")
            _store = _GCS(bucket) if bucket else _Local()
    return _store


def wav_hash(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


class SampleCache:
    """Cache for one sample call under the current pipeline version and ASR mode."""

    def __init__(self, sample_id, wav_path, version, asr):
        self.sample_id = sample_id
        self.ns = f"{EPOCH}/{version}-{asr}/{sample_id}-{wav_hash(wav_path)}"
        self.st = store()

    def _put(self, key, data: bytes):
        try:
            self.st.put(key, data)
        except Exception as e:  # noqa: BLE001  (a failed write must never break the call; the next play retries)
            print("sample cache write failed:", key, str(e)[:120], flush=True)

    def _get(self, key):
        try:
            return self.st.get(key)
        except Exception as e:  # noqa: BLE001
            print("sample cache read failed:", key, str(e)[:120], flush=True)
            return None

    # ---- speech-to-text trace ----
    def get_stt(self):
        raw = self._get(f"{self.ns}/stt.json")
        return json.loads(raw) if raw else None

    def put_stt(self, events):
        self._put(f"{self.ns}/stt.json", json.dumps({"saved": time.time(), "events": events}).encode())

    # ---- cards ----
    def _card_key(self, idx, text):
        return f"{self.ns}/cards/{idx:03d}-{_h(text)}.json"

    def get_card(self, idx, text):
        raw = self._get(self._card_key(idx, text))
        return json.loads(raw) if raw else None

    def put_card(self, idx, text, card, update=None):
        self._put(self._card_key(idx, text), json.dumps({"card": card, "update": update}, default=float).encode())

    # ---- calm voice ----
    def _voice_key(self, text):
        return f"{self.ns}/voice/{_h(text)}.pcm"

    def get_voice(self, text):
        raw = self._get(self._voice_key(text))
        return np.frombuffer(raw, "<i2").astype(np.float32) / 32767 if raw else None

    def put_voice(self, text, pcm: np.ndarray):
        self._put(self._voice_key(text), (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())


def status():
    """Per-namespace contents, for the admin page and the demo's 'cached' badges."""
    out = {}
    for key, size, mtime in store().list(f"{EPOCH}/"):
        parts = key.split("/")
        if len(parts) < 4:
            continue
        ns = "/".join(parts[1:3])
        e = out.setdefault(ns, {"config": parts[1], "sample": parts[2].rsplit("-", 1)[0], "stt": False, "cards": 0,
                                "voice": 0, "bytes": 0, "updated": 0})
        e["bytes"] += size; e["updated"] = max(e["updated"], mtime)
        if parts[3] == "stt.json":
            e["stt"] = True
        elif parts[3] == "cards":
            e["cards"] += 1
        elif parts[3] == "voice":
            e["voice"] += 1
    return {"backend": store().name, "epoch": EPOCH, "entries": sorted(out.values(), key=lambda e: (e["config"], e["sample"]))}


def clear(sample_id=None):
    """Delete everything (all epochs and configs), or one sample's entries in every config."""
    if sample_id is None:
        return store().delete_prefix("")
    keys = [k.split("/") for k, _, _ in store().list("")]
    namespaces = {"/".join(p[:3]) for p in keys if len(p) >= 4 and p[2].rsplit("-", 1)[0] == sample_id}
    return sum(store().delete_prefix(ns + "/") for ns in namespaces)
