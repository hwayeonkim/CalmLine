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
# Our own endpointing: Google's endpointer waits for long silences (and TTS-voiced or fast talkers barely pause), so a
# finished sentence could wait 5-30 s for its final. After PAUSE_S of non-speech we half-close the stream, which makes
# Google finalize what it has at once, and continue on a fresh stream - the same 0.45 s pause rule as the Whisper path.
PAUSE_S = float(os.getenv("CALMLINE_GOOGLE_PAUSE_S", "0.45"))
VOICE_DB = -45.0              # 10 ms frames above this level count as speech (same threshold as the session gauge)


def available() -> tuple[bool, str]:
    try:
        import google.auth
        from google.cloud import speech_v1  # noqa: F401
        creds, project = google.auth.default()
        return True, f"ADC ok (project {project})"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:160]


def _voiced_frames(x: np.ndarray) -> np.ndarray:
    """Indices of 25 ms frames (10 ms hop) whose level is above VOICE_DB."""
    if len(x) < 400:
        return np.zeros(0, int)
    fr = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), 400)[::160]
    db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    return np.nonzero(db > VOICE_DB)[0]


class GoogleStreamingSTT:
    """Feed 16 kHz float32 chunks with feed(); collect events with poll().

    Events:
      ("final",   t0, t1, text, words)   one finished part (times in seconds of call audio)
      ("interim", t_end, text, stability)
    """

    def __init__(self, language="en-US", model=None, phrase_hints=None, pause_s=None):
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
        self.q_in: queue.Queue | None = None     # current stream's audio queue; None until audio arrives
        self.thread = None
        self.q_out: queue.Queue = queue.Queue()
        self.offset = 0.0           # call time at which the current stream started
        self.sent = 0.0             # seconds of audio sent in the current stream
        self.last_final_end = 0.0
        self.closed = False
        self.error = None
        self.pause_s = PAUSE_S if pause_s is None else pause_s
        self.fed = 0.0              # call time fed so far
        self.last_voice = 0.0       # call time of the last speech frame
        self.voiced = False         # speech sent to the current stream since it started

    # ---- streaming thread ----
    def _requests(self, q):
        # bound to this stream's queue: after a cut, self.q_in already points at the next stream's queue
        while True:
            b = q.get()
            if b is None:
                return
            yield self.sp.StreamingRecognizeRequest(audio_content=b)

    def _run(self, offset, q):
        try:
            responses = self.client.streaming_recognize(config=self.stream_cfg, requests=self._requests(q))
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
        # streams open lazily on the first audio after a cut, so a paused call never holds an idle stream open
        # (Google drops a stream that gets no audio for ~10 s), and offset is the call time the stream starts at
        self.offset = self.fed
        self.q_in = queue.Queue()
        self.sent = 0.0
        self.voiced = False
        self.thread = threading.Thread(target=self._run, args=(self.offset, self.q_in), daemon=True)
        self.thread.start()

    # ---- public ----
    def feed(self, chunk: np.ndarray):
        if self.closed:
            return
        if self.sent > STREAM_LIMIT_S:            # roll over to a fresh stream before Google's limit
            self.cut()
        if self.q_in is None or not self.thread.is_alive():
            self._start()           # first audio after a cut, or the stream ended on an error: open a new one
        self.q_in.put((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
        self.sent += len(chunk) / SR
        voiced = _voiced_frames(chunk)
        if len(voiced):
            self.voiced = True
            self.last_voice = self.fed + (voiced[-1] * 160 + 400) / SR
        self.fed += len(chunk) / SR
        if self.voiced and self.fed - self.last_voice >= self.pause_s:
            self.cut()                            # the caller paused: finalize this part now

    def cut(self):
        """Half-close the current stream (Google returns its final right away); the next audio opens a new one."""
        if self.q_in is not None:
            self.q_in.put(None)
            self.q_in = None

    def close(self, wait=3.0):
        self.closed = True
        self.cut()
        if self.thread is not None:
            self.thread.join(timeout=wait)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q_out.get_nowait())
            except queue.Empty:
                return out
