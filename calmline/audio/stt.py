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
    return WhisperModel(size, device=device, compute_type=compute_type, cpu_threads=8)


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


class WindowSpotter:
    """Constant-cost first pass for *mute decisions* (two-pass design, cf. Sainath et al., 2019).

    LocalAgreement re-decodes the whole growing utterance and waits for two agreeing hypotheses, which on CPU
    makes it ~real-time-factor 1 and too slow for a 2 s buffer. For muting we only need local context, so every
    hop we decode just the last `window_s` seconds with a small model and act on that hypothesis immediately.
    The calm card still uses a full-context second pass at the end of the utterance."""

    PROFANITY_PROMPT = ("Verbatim, uncensored transcript of an angry phone call to a city hotline. "
                        "Fuck, fucking, shit, bitch, asshole, idiot, moron, dumbass, bastard, goddamn.")

    def __init__(self, size="tiny.en", window_s=3.0, bias_prompt=True):
        self.model = load_whisper(size)
        self.window_s = window_s
        # Whisper under-transcribes profanity; a vocabulary prompt raised toxic-token recognition
        # 92.0% -> 95.6% (base.en) with 0/30 hallucinated profanity on clean calls (results/audio/whisper_prompt_bias.json)
        self.prompt = self.PROFANITY_PROMPT if bias_prompt else None
        self.reset()

    def reset(self):
        self.audio = np.zeros(0, np.float32)

    def feed(self, chunk: np.ndarray, final=False) -> list[Word]:
        self.audio = np.concatenate([self.audio, chunk.astype(np.float32)])
        start = max(0.0, len(self.audio) / SR - self.window_s)
        seg = self.audio[int(start * SR):]
        # silence gate: a vocabulary prompt on near-silent audio makes Whisper echo the prompt (seen in the demo)
        from .acoustic import frame_rms
        db = 20 * np.log10(frame_rms(seg) + 1e-9)
        if (db > -45).mean() * len(seg) / SR < 0.3:
            return []
        segs, _ = self.model.transcribe(seg, language="en", beam_size=1, word_timestamps=True, initial_prompt=self.prompt,
                                        condition_on_previous_text=False, vad_filter=False, without_timestamps=False)
        out = [Word(w.start + start, w.end + start, w.word.strip()) for s in segs for w in (s.words or []) if w.word.strip()]
        if self.prompt and len(out) >= 4 and " ".join(w.text for w in out).lower().strip(" .") in self.prompt.lower():
            return []   # prompt echo
        return out


def normalize_asr(text: str) -> str:
    """Undo Whisper formatting that breaks slot extraction: '1,820' -> '1820', '311, 20871455' -> '311-20871455'."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\b311[,.]?\s+(\d{6,10})\b", r"311-\1", text)
    text = re.sub(r"\b(\d{1,3})\s*-\s*(\d{2})\b", r"\1-\2", text)
    return text
