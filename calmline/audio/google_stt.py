"""Google Cloud Speech-to-Text v2 streaming adapter (production path on Cloud Run).

Same interface as StreamingWhisper: feed(chunk) -> newly *final* words. Google's streaming API
returns interim hypotheses (`is_final=False`) and stable final results; we treat final results
as "committed" (equivalent to LocalAgreement commits) and expose the latest interim result as
`prev` for the low-latency 'tentative' policy.

Requires: `pip install google-cloud-speech`, ADC credentials, and a recognizer location that
supports the chosen model (e.g. model="chirp_2" or "long"/"telephony" in a supported region).
NOTE: written against the v2 API surface but not exercised in this repo's offline experiments
(no GCP credentials were available while building the prototype)."""
from __future__ import annotations
import queue
import threading

import numpy as np

from .stt import Word


class GoogleStreamingSTT:
    def __init__(self, project: str, location: str = "us", model: str = "telephony", language="en-US"):
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech as cs
        from google.api_core.client_options import ClientOptions
        opts = ClientOptions(api_endpoint=f"{location}-speech.googleapis.com") if location != "global" else None
        self.client = SpeechClient(client_options=opts)
        self.cs = cs
        self.recognizer = f"projects/{project}/locations/{location}/recognizers/_"
        self.config = cs.StreamingRecognitionConfig(
            config=cs.RecognitionConfig(
                explicit_decoding_config=cs.ExplicitDecodingConfig(encoding=cs.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                                                                   sample_rate_hertz=16000, audio_channel_count=1),
                language_codes=[language], model=model,
                features=cs.RecognitionFeatures(enable_word_time_offsets=True, enable_automatic_punctuation=True,
                                                profanity_filter=False)),   # we need the profanity to detect it
            streaming_features=cs.StreamingRecognitionFeatures(interim_results=True))
        self.q: queue.Queue = queue.Queue()
        self.committed: list[Word] = []
        self.prev: list[Word] = []
        self._new: list[Word] = []
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _requests(self):
        yield self.cs.StreamingRecognizeRequest(recognizer=self.recognizer, streaming_config=self.config)
        while True:
            b = self.q.get()
            if b is None:
                return
            yield self.cs.StreamingRecognizeRequest(audio=b)

    def _run(self):
        for resp in self.client.streaming_recognize(requests=self._requests()):
            for res in resp.results:
                alt = res.alternatives[0] if res.alternatives else None
                if alt is None:
                    continue
                words = [Word(w.start_offset.total_seconds(), w.end_offset.total_seconds(), w.word) for w in alt.words] \
                    or [Word(0, 0, t) for t in alt.transcript.split()]
                with self._lock:
                    if res.is_final:
                        self.committed += words
                        self._new += words
                        self.prev = []
                    else:
                        self.prev = words

    def feed(self, chunk: np.ndarray, final=False):
        self.q.put((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
        if final:
            self.q.put(None)
            self._t.join(timeout=10)
        with self._lock:
            new, self._new = self._new, []
        return new

    def text(self):
        return " ".join(w.text for w in self.committed)
