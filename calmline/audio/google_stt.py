"""Google Cloud Speech-to-Text streaming (v1 StreamingRecognize) for the live relay.

Replaces both local passes: Google's streaming endpointer closes a part (is_final) shortly after the caller pauses,
and the final result already carries the transcript and word offsets — no separate end-of-sentence heuristic and no
second Whisper pass. Interim results (with stability) are exposed for the Shield mode's mute decisions.

Auth: Speech-to-Text does not accept API keys. Use Application Default Credentials
(`gcloud auth application-default login`) or a service-account key via GOOGLE_APPLICATION_CREDENTIALS, on a project
with the Speech-to-Text API enabled.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np

SR = 16000
STREAM_LIMIT_S = 280          # Google closes a single stream at ~5 minutes; we roll over before that


def available() -> tuple[bool, str]:
    try:
        import google.auth
        from google.cloud import speech_v1  # noqa: F401
        creds, project = google.auth.default()
        return True, f"ADC ok (project {project})"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:160]


class GoogleStreamingSTT:
    """Feed 16 kHz float32 chunks with feed(); collect events with poll().

    Events:
      ("final",   t0, t1, text, words)   one finished part (times in seconds of call audio)
      ("interim", t_end, text, stability)
    """

    def __init__(self, language="en-US", model=None, phrase_hints=None):
        from google.cloud import speech_v1 as sp
        self.sp = sp
        self.client = sp.SpeechClient()
        self.model = model or os.getenv("CALMLINE_GOOGLE_STT_MODEL", "latest_short")
        cfg = sp.RecognitionConfig(
            encoding=sp.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=SR, language_code=language,
            model=self.model, use_enhanced=True, enable_automatic_punctuation=True, enable_word_time_offsets=True,
            profanity_filter=False,                       # we must see the abuse to remove it
            speech_contexts=[sp.SpeechContext(phrases=phrase_hints or [], boost=8.0)] if phrase_hints else None)
        self.stream_cfg = sp.StreamingRecognitionConfig(config=cfg, interim_results=True, single_utterance=False)
        self.q_in: queue.Queue = queue.Queue()
        self.q_out: queue.Queue = queue.Queue()
        self.offset = 0.0           # call time at which the current stream started
        self.sent = 0.0             # seconds of audio sent in the current stream
        self.last_final_end = 0.0
        self.closed = False
        self.error = None
        self._start()

    # ---- streaming thread ----
    def _requests(self):
        while True:
            b = self.q_in.get()
            if b is None:
                return
            yield self.sp.StreamingRecognizeRequest(audio_content=b)

    def _run(self, offset):
        try:
            responses = self.client.streaming_recognize(config=self.stream_cfg, requests=self._requests())
            for resp in responses:
                for res in resp.results:
                    if not res.alternatives:
                        continue
                    alt = res.alternatives[0]
                    t_end = offset + res.result_end_time.total_seconds()
                    if res.is_final:
                        words = [(offset + w.start_time.total_seconds(), offset + w.end_time.total_seconds(), w.word) for w in alt.words]
                        t0 = words[0][0] if words else self.last_final_end
                        self.last_final_end = t_end
                        self.q_out.put(("final", t0, t_end, alt.transcript.strip(), words))
                    else:
                        self.q_out.put(("interim", t_end, alt.transcript.strip(), res.stability))
        except Exception as e:  # noqa: BLE001
            self.error = str(e)[:200]
            self.q_out.put(("error", self.error))

    def _start(self):
        self.q_in = queue.Queue()
        self.sent = 0.0
        self.thread = threading.Thread(target=self._run, args=(self.offset,), daemon=True)
        self.thread.start()

    # ---- public ----
    def feed(self, chunk: np.ndarray):
        if self.closed:
            return
        if self.sent > STREAM_LIMIT_S:            # roll over to a fresh stream before Google's limit
            self.q_in.put(None)
            self.offset += self.sent
            self._start()
        self.q_in.put((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
        self.sent += len(chunk) / SR

    def close(self, wait=3.0):
        self.closed = True
        self.q_in.put(None)
        self.thread.join(timeout=wait)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q_out.get_nowait())
            except queue.Empty:
                return out
