"""Live call session: delay buffer + streaming STT + abuse gate + utterance pipeline.

Audio path (caller -> agent):
    caller PCM ──> ring buffer ──(D seconds later)──> [mute mask] ─> [look-ahead compressor] ─> agent
                         │
                         └─> StreamingWhisper ─> committed words ─> detector ─> mute intervals
                                                                    └─> utterance end ─> Pipeline ─> calm card
"""
from __future__ import annotations
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from .audio.acoustic import LookaheadCompressor, frame_rms

SR = 16000
import re
CLAUSE_STARTERS = {"and", "but", "so", "because", "cause", "which", "who", "then", "or", "since", "while", "when", "if", "anyway", "plus"}


@dataclass
class Settings:
    delay_s: float = 2.0
    chunk_s: float = 0.5
    protection: str = "calm_voice"  # calm_voice (default: agent hears only the calm Gemini voice) | shield | raw
    compress: bool = True
    whisper: str = "base.en"        # pass 2 (utterance card); small.en cost ~1 s per sentence on CPU
    spotter: str = "tiny.en"        # pass 1 (mute decisions)
    pause_s: float = 0.45           # pause that closes a sentence-level utterance
    chunking: str = "clause"        # "sentence" (v17) or "clause" (v18: syntax/prosody-aware sub-sentence parts)


class CallSession:
    def __init__(self, pipeline, detector, settings: Settings | None = None, evidence_dir="evidence"):
        self.pipe, self.det = pipeline, detector
        self.s = settings or Settings()
        self.id = uuid.uuid4().hex[:8]
        self.audio = np.zeros(0, np.float32)       # all caller audio (sealed evidence copy lives on disk)
        self.mutes: list[tuple[float, float, str]] = []
        self.cards: list[dict] = []
        self.events: asyncio.Queue = asyncio.Queue()
        self.utt_start = None
        self.last_voice = 0.0
        self.alerted = False
        self.flagged_word_ids: set = set()
        self.anger = 0.0
        self.baseline_db = None
        self.comp = LookaheadCompressor()
        self.evidence_dir = os.path.join(evidence_dir, self.id)
        os.makedirs(self.evidence_dir, exist_ok=True)
        self._stt = None
        self.t0 = time.time()
        self.closed = False
        self.audit = []
        self.voice_log: list[tuple[float, np.ndarray]] = []   # (seconds since call start, 16 kHz chunk) of the calm voice

    @property
    def stt(self):
        if self._stt is None:
            from .audio.stt import WindowSpotter
            self._stt = WindowSpotter(self.s.spotter, window_s=3.0, bias_prompt=False)
        return self._stt

    # ---------------- ingest ----------------
    def ingest(self, pcm: np.ndarray, final=False):
        """Called (in a worker thread) for every caller chunk. Returns list of events.

        Two-pass design: pass 1 (small model, last 3 s window) decides mutes immediately;
        pass 2 (larger model, whole utterance) produces the calm card when a pause ends the utterance."""
        start = len(self.audio) / SR
        self.audio = np.concatenate([self.audio, pcm.astype(np.float32)])
        ev = []
        db = 20 * np.log10(frame_rms(pcm).mean() + 1e-9)
        voiced = db > -45
        if voiced:
            self.baseline_db = db if self.baseline_db is None else 0.97 * self.baseline_db + 0.03 * min(db, self.baseline_db + 3)
            rel = db - self.baseline_db
            self.anger = float(np.clip(0.8 * self.anger + 0.2 * np.clip(rel / 12, 0, 1), 0, 1))
            self.last_voice = start + len(pcm) / SR
            if self.utt_start is None:
                self.utt_start = start
        # pass 1: mute decisions on the sliding window
        words = self.stt.feed(pcm)
        text, offs = "", []
        for w in words:
            if text: text += " "
            offs.append((len(text), len(text) + len(w.text))); text += w.text
        r = self.det(text) if text else None
        if r is not None:
            for sp in r.spans:
                for i, (a, b) in enumerate(offs):
                    if b <= sp.start or a >= sp.end:
                        continue
                    w = words[i]
                    key = (round(w.start, 1), w.text.lower())
                    if any(abs(k[0] - key[0]) < 0.3 and k[1] == key[1] for k in self.flagged_word_ids):
                        continue
                    self.flagged_word_ids.add(key)
                    self.mutes.append((w.start - 0.25, w.end + 0.25, sp.label))
                    ev.append({"type": "mute", "start": w.start, "end": w.end, "label": sp.label})
                    # no alert from pass 1: a 3-second window of tiny.en is too noisy to page a supervisor on;
                    # the alert is raised from the verified pass-2 sentence (process_utt) instead.
        ev.append({"type": "partial", "anger": round(self.anger, 2), "ct": round(start + len(pcm) / SR, 3),
                   "c0": round(start, 3), "voiced": bool(voiced)})
        # pass 2 is scheduled, not run here, so mute decisions on the next chunks are never blocked.
        # Utterance boundary = a short pause (sentence-level relay) or end of call.
        now = start + len(pcm) / SR
        # sentence-level relay without waiting for a pause: if pass 1 has seen a sentence end (. ? !) at least
        # 0.3 s ago and 1.5 s into the utterance, cut there so long rants are relayed sentence by sentence.
        if self.utt_start is not None and not final and words:
            cands = []
            for i, w in enumerate(words):
                nxt_w = words[i + 1] if i + 1 < len(words) else None
                nxt = nxt_w.start if nxt_w else w.end + 0.4
                if self.s.chunking == "clause":
                    # v18 clause-level parts (SASST-style syntactic completeness, approximated with punctuation and
                    # conjunctions; Bangalore et al. 2012 prosodic pauses). wait-1: the boundary is only taken once the
                    # next word has been heard, and a part must be >= 1.0 s and >= 4 words so it can be judged on its own.
                    in_utt = [x for x in words if x.start >= self.utt_start - 0.05 and x.end <= w.end + 0.01]
                    if nxt_w is None or len(in_utt) < 4 or w.end < self.utt_start + 1.0 or now - w.end < 0.25:
                        continue
                    punct = w.text.endswith((".", "?", "!", ",", ";", ":"))
                    conj = re.sub(r"[^a-z']", "", nxt_w.text.lower()) in CLAUSE_STARTERS
                    pause = nxt - w.end >= 0.25
                    if not (punct or conj or pause):
                        continue
                    if not pause and nxt - w.end < 0.05 and not w.text.endswith((".", "?", "!")):
                        continue      # glued words: likely mid-phrase
                    cands.append((w.end, nxt))
                    continue
                if not w.text.endswith((".", "?", "!")) or w.end < self.utt_start + 1.5 or now - w.end < 0.3:
                    continue
                if nxt - w.end < 0.08 and i + 1 < len(words):
                    continue          # no audible gap: Whisper punctuation, not a real sentence break
                cands.append((w.end, nxt))
            if cands:
                a, b = cands[-1]
                cut = self._quietest(a - 0.12, max(b, a + 0.12))
                ev.append({"type": "utt", "t0": self.utt_start, "t1": cut})
                self.utt_start = cut
                return ev
        if self.utt_start is not None and (final or now - self.last_voice >= self.s.pause_s):
            ev.append({"type": "utt", "t0": self.utt_start, "t1": self.last_voice if not final else now})
            self.utt_start = None
        return ev

    def _quietest(self, a, b):
        """Time of the lowest-energy 20 ms frame in [a, b] — where a sentence cut is least likely to split a word."""
        i, j = int(max(0, a) * SR), int(b * SR)
        x = self.audio[i:j]
        if len(x) < 640:
            return (a + b) / 2
        e = frame_rms(x, 320, 160)
        return a + float(np.argmin(e)) * 160 / SR + 0.01

    def process_utterance(self, t0, t1, idx=None):
        return self._close_utterance(t0, t1, idx)

    def process_text(self, text):
        """Typed caller text (judges testing without a microphone): same pipeline, no ASR."""
        return self._card(text, [0.0, 0.0], asr_ms=0, idx=getattr(self, "_typed_seq", 0))

    def _close_utterance(self, t0, t1, idx=None):
        import time as _t
        _ts = _t.perf_counter()
        from .audio.stt import load_whisper
        x = self.audio[int(max(0, t0) * SR):int((t1 + 0.05) * SR)]
        segs, _ = load_whisper(self.s.whisper).transcribe(x, language="en", beam_size=1, initial_prompt=self.stt.PROFANITY_PROMPT,
                                                          condition_on_previous_text=False)
        from .audio.stt import normalize_asr
        text = normalize_asr(" ".join(s.text.strip() for s in segs).strip())
        if not text:
            return None
        _asr = (_t.perf_counter() - _ts) * 1000
        return self._card(text, [t0, t1], _asr, idx)

    def _card(self, text, t, asr_ms, idx=None):
        import time as _t
        _ts = _t.perf_counter()
        gen = getattr(getattr(self.pipe, "renderer", None), "gen", None)
        if gen is not None and hasattr(gen, "context"):
            gen.context = " ".join(c["delivered"].split("] ", 1)[-1] for c in self.cards[-3:])
        if hasattr(self.pipe, "context"):
            self.pipe.context = " ".join(c["delivered"].split("] ", 1)[-1] for c in self.cards[-4:] if c["delivered"])
        d = self.pipe(text)
        _pipe = (_t.perf_counter() - _ts) * 1000
        from .intents import INTENTS_ALL as INTENTS
        u = d.debug.get("understanding") or {}
        spans = list(d.debug.get("spans", []))
        for ph in u.get("abusive_phrases", []):
            i = text.lower().find(ph.lower())
            if i >= 0 and not any(a <= i < b for a, b, _ in spans):
                spans.append((i, i + len(ph), "gemini"))
        checks = {}
        if d.route != "PASS" and d.delivered.split("] ", 1)[-1].strip():
            from .textnorm import numbers
            out = d.delivered.split("] ", 1)[-1]
            checks = {"no abuse left": not self.det(out).is_toxic, "no new numbers": not (numbers(out) - numbers(text) - {"311"}),
                      "fallback used": d.tier == 3, "rejected": d.debug.get("rejected")}
        late = d.debug.get("late_future")
        idx = len(self.cards) if idx is None else idx
        if late is not None:
            self.late = getattr(self, "late", {}); self.late[idx] = late
        card = dict(type="card", idx=idx, original=text, spans=sorted(spans),
                    path=d.debug.get("path") or ("Gemini" if u else "offline"),
                    intent_label=INTENTS[d.intent]["issue"] if d.intent in INTENTS else None,
                    nyc_types=INTENTS[d.intent]["nyc"] if d.intent in INTENTS else [], intent_score=round(d.intent_score, 2),
                    slots_dropped=u.get("slots_dropped", []), checks=checks, model=u.get("model"),
                    route=d.route, delivered=d.delivered, withheld=(d.route != "PASS" and not d.delivered.split("] ", 1)[-1].strip()), intent=d.intent, slots=d.slots, tier=d.tier,
                    abuse_types=d.abuse_types, anger=round(max(self.anger, d.anger if d.route != "PASS" else 0), 2),
                    t=t, latency_ms=round(sum(d.timings_ms.values()), 1),
                    timing=dict(wait_for_pause_s=self.s.pause_s, asr_ms=round(asr_ms), rewrite_pipeline_ms=round(_pipe),
                                **{k: round(v) for k, v in d.timings_ms.items()}))
        card_public = dict(card)
        # sealed original: kept for evidence, never sent to the agent UI
        with open(os.path.join(self.evidence_dir, "transcript.jsonl"), "a") as f:
            f.write(json.dumps({"t": card["t"], "original": text, "route": d.route, "abuse_types": d.abuse_types}) + "\n")
        self.voice_by_idx = getattr(self, "voice_by_idx", {})
        self.cards.append(card_public)
        return card_public

    # ---------------- playback ----------------
    def render_playback(self, t_from: float, t_to: float) -> np.ndarray:
        """Audio the agent hears for caller-time [t_from, t_to) (already delayed by the caller)."""
        a, b = int(t_from * SR), int(t_to * SR)
        x = self.audio[a:b].copy()
        if len(x) == 0:
            return x
        if self.s.protection == "calm_voice":
            return np.zeros_like(x)         # agent hears only the calm rendering (spoken by browser TTS)
        if self.s.protection == "raw":
            return x
        tt = np.arange(a, a + len(x)) / SR
        for ms, me, _ in self.mutes:
            sel = (tt >= ms) & (tt < me)
            if sel.any():
                x[sel] = 0.004 * np.sin(2 * np.pi * 440 * tt[sel])   # soft 'masked' tone, not silence, so the agent knows
        if self.s.compress:
            ctx = self.audio[max(0, a - SR // 2):b + SR // 2]
            g = self.comp.gain_curve(ctx)
            off = a - max(0, a - SR // 2)
            x = x * g[off:off + len(x)]
        return x.astype(np.float32)

    def save_evidence(self):
        import soundfile as sf
        sf.write(os.path.join(self.evidence_dir, "caller_original.wav"), self.audio, SR)
        # what the agent heard, saved for the demo's before/after comparison
        if self.s.protection == "calm_voice" and self.voice_log:
            end = 0.0
            heard = np.zeros(int((len(self.audio) / SR + 30) * SR), np.float32)
            for t, x in self.voice_log:
                st = max(t, end); a = int(st * SR)
                if a + len(x) > len(heard):
                    heard = np.concatenate([heard, np.zeros(a + len(x) - len(heard), np.float32)])
                heard[a:a + len(x)] += x; end = st + len(x) / SR
            heard = heard[:int((end + 0.5) * SR)]
        else:
            prot = self.s.protection
            self.s.protection = "shield" if prot == "calm_voice" else prot
            heard = self.render_playback(0.0, len(self.audio) / SR)
            self.s.protection = prot
        sf.write(os.path.join(self.evidence_dir, "agent_heard.wav"), heard, SR)
        json.dump(dict(mutes=self.mutes, cards=self.cards, audit=self.audit), open(os.path.join(self.evidence_dir, "session.json"), "w"), indent=1)
