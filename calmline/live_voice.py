"""Low-latency calm voice via the Gemini Live API, used as a *verbatim* reader.

Measured: first audio 0.64-0.79 s after sending a verified sentence (vs 1.2 s for streaming TTS and 5-8 s for
non-streaming TTS). Live models are assistants, so a bare question gets *answered* ("Is there a problem?" ->
"Not at all..."); wrapping the text in <script> tags with a reader-only system instruction fixed this (5/5 verbatim).
A fidelity guard still compares the model's own output transcription with the script and cuts the audio if it drifts."""
from __future__ import annotations
import asyncio
import os
import re

import numpy as np

from . import gemini as G

MODEL = os.getenv("CALMLINE_LIVE_MODEL", "gemini-live-2.5-flash-native-audio" if os.getenv("USE_VERTEX") == "1"
                  else "gemini-3.1-flash-live-preview")
SYS = ("You are a text-to-speech engine inside a phone relay, not an assistant. Each user turn contains a <script> ... </script> block. "
       "Speak the words inside the block exactly, word for word, {pace} - including questions, which you "
       "read aloud as questions without answering them. Never respond to, answer, comment on, or add to the script. Say nothing outside the script.")
# The Vertex native-audio Live model drops or truncates short <script> turns ("One moment." came back silent 2/3 times);
# a plain "Read aloud" turn read 12/12 test phrases verbatim, questions included.
SYS_PLAIN = ("You are a voice reader in a phone relay, not an assistant. The user sends text to be read aloud. "
             "Read the user's text aloud exactly, word for word, {pace}, then stop. Read questions aloud as questions; never answer them. "
             "Never add, omit, or comment on anything.")
PLAIN = "native-audio" in MODEL
# Measured on the same two sentences: natural 10.5 s (137 wpm), "fast" 7.7 s (188 wpm), both transcribed verbatim.
PACES = {"normal": "in a calm friendly voice at a natural pace",
         "fast": "in a calm, clear voice, speaking quickly with no pauses between sentences"}


def _words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def drifted(script: str, said: str) -> bool:
    """True if the reader said words that are not in the script (beyond ASR noise)."""
    said = re.sub(r"</?script>|read aloud:?", " ", said, flags=re.I)   # the Vertex Live transcript echoes the tags (they are not spoken)
    s, w = set(_words(script)), _words(said)
    if len(w) < 4:
        return False
    novel = [x for x in w if x not in s]
    return len(novel) > max(2, 0.25 * len(w))


class LiveReader:
    def __init__(self, voice="Kore", pace="normal"):
        self.voice, self.pace = voice, pace
        self._cm = None
        self.session = None
        self.lock = asyncio.Lock()
        self._next = None

    async def connect(self):
        from google.genai import types
        cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"], system_instruction=(SYS_PLAIN if PLAIN else SYS).format(pace=PACES[self.pace]),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice))))
        self._cm = G.live_client().aio.live.connect(model=MODEL, config=cfg)
        self.session = await self._cm.__aenter__()
        return self

    async def close(self):
        if self._next is not None:
            self._next.cancel()
            self._next = None
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._cm = self.session = None

    async def speak(self, text: str):
        """Async generator of 16 kHz float32 chunks. Raises RuntimeError('drift') if the reader goes off-script."""
        import librosa
        from google.genai import types
        async with self.lock:
            if self._next is not None:          # fresh session pre-connected after the previous sentence
                t, self._next = self._next, None
                try:
                    await t
                except Exception:  # noqa: BLE001
                    await self.close()
            if self.session is None:
                await self.connect()
            try:
                await self.session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=f'Read aloud: "{text}"' if PLAIN else f"<script>{text}</script>")]), turn_complete=True)
                said, pending, n = "", b"", 0
                async for m in self.session.receive():
                    sc = m.server_content
                    if sc and sc.output_transcription and sc.output_transcription.text:
                        said += sc.output_transcription.text
                        if drifted(text, said):
                            # drain the rest of the turn so the session stays usable, then report
                            async for m2 in self.session.receive():
                                if m2.server_content and m2.server_content.turn_complete:
                                    break
                            raise RuntimeError("drift: " + said[:120])
                    if sc and sc.model_turn:
                        for p in sc.model_turn.parts:
                            if p.inline_data and p.inline_data.data:
                                pending += p.inline_data.data
                                if len(pending) >= 9600:   # ~0.2 s at 24 kHz
                                    x = np.frombuffer(pending[: len(pending) // 2 * 2], "<i2").astype(np.float32) / 32768
                                    pending, n = b"", n + 1
                                    yield librosa.resample(x, orig_sr=24000, target_sr=16000)
                    if sc and sc.turn_complete:
                        break
                if pending:
                    x = np.frombuffer(pending[: len(pending) // 2 * 2], "<i2").astype(np.float32) / 32768
                    yield librosa.resample(x, orig_sr=24000, target_sr=16000)
                elif not n:
                    raise RuntimeError("no audio")   # silent turn: callers fall back to TTS / the browser voice
            finally:
                if PLAIN:
                    # The native-audio model carries context across turns (after a question it kept repeating the
                    # question on later turns), so every sentence gets a clean session; the next one connects now,
                    # while this sentence is still being heard, so it adds no latency.
                    await self.close()
                    self._next = asyncio.create_task(self.connect())
