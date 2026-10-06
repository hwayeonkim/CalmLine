"""Streaming STT: faster-whisper + LocalAgreement-2 commit policy.

Whisper is not a streaming model. Following Macháček, Dabre & Bojar (2023), "Turning Whisper into
Real-Time Transcription System", we re-transcribe the growing audio buffer every chunk and only
*commit* the longest common prefix of two consecutive hypotheses (LocalAgreement-n, n=2). Committed
words are stable, so the abuse detector can act on them without later retractions."""
from __future__ import annotations
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

SR = 16000


@dataclass
class Word:
    start: float
    end: float
    text: str


@lru_cache(maxsize=4)
def load_whisper(size="base.en", device="cpu", compute_type="int8"):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device=device, compute_type=compute_type)


def _norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


class StreamingWhisper:
    def __init__(self, size="base.en", trim_s=12.0, beam_size=1):
        self.model = load_whisper(size)
        self.trim_s, self.beam = trim_s, beam_size
        self.reset()

    def reset(self):
        self.audio = np.zeros(0, np.float32)
        self.offset = 0.0              # absolute time of self.audio[0]
        self.committed: list[Word] = []
        self.prev: list[Word] = []     # previous uncommitted hypothesis tail

    def _transcribe(self, final=False):
        prompt = " ".join(w.text for w in self.committed[-30:])
        segs, _ = self.model.transcribe(self.audio, language="en", beam_size=self.beam, word_timestamps=True,
                                        initial_prompt=prompt or None, condition_on_previous_text=False, vad_filter=False)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append(Word(w.start + self.offset, w.end + self.offset, w.word.strip()))
        last = self.committed[-1].end if self.committed else -1
        words = [w for w in words if w.start >= last - 0.6 and w.text]
        # remove n-gram overlap with the committed tail (whisper_streaming HypothesisBuffer.insert)
        tail = [_norm(w.text) for w in self.committed[-5:]]
        for n in range(min(5, len(tail), len(words)), 0, -1):
            if tail[-n:] == [_norm(w.text) for w in words[:n]]:
                words = words[n:]
                break
        return words

    def feed(self, chunk: np.ndarray, final=False) -> list[Word]:
        """Append audio, return newly committed words."""
        self.audio = np.concatenate([self.audio, chunk.astype(np.float32)])
        hyp = self._transcribe(final)
        if final:
            new = hyp
        else:
            new = []
            for a, b in zip(self.prev, hyp):
                if _norm(a.text) == _norm(b.text) and _norm(a.text):
                    new.append(b)
                else:
                    break
        self.committed += new
        self.prev = hyp[len(new):] if not final else []
        # trim buffer at committed boundary
        if self.committed and (len(self.audio) / SR) > self.trim_s:
            cut = self.committed[-1].end - self.offset
            k = int(cut * SR)
            if 0 < k < len(self.audio):
                self.audio = self.audio[k:]
                self.offset += cut
        return new

    def text(self):
        return " ".join(w.text for w in self.committed)


def offline_words(path_or_audio, size="small.en"):
    import soundfile as sf
    x = sf.read(path_or_audio)[0].astype(np.float32) if isinstance(path_or_audio, str) else path_or_audio
    segs, _ = load_whisper(size).transcribe(x, language="en", beam_size=5, word_timestamps=True)
    return [Word(w.start, w.end, w.word.strip()) for s in segs for w in (s.words or [])]
