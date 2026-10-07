"""Google text-to-speech for the Calm-voice relay.

Gemini TTS (streaming; gemini-3.8-flash-lite-tts gives first audio in ~1.2 s) -> Google Cloud TTS -> none.
Audio is cached by text, so replayed demo calls are instant and cost no quota."""
from __future__ import annotations
import io
import os
import re

import numpy as np

from . import gemini as G

VOICE = os.getenv("CALMLINE_VOICE", "Kore")
STYLE = "calm, friendly, natural conversational pace"


def prompt(text: str, style: str) -> str:
    # "Say <style>: <text>" made Gemini TTS read the instruction aloud (verified with Whisper);
    # the director's-notes / transcript layout keeps the style out of the audio.
    return f"### DIRECTOR'S NOTES\nStyle: {style}.\n\n### TRANSCRIPT\n{text}"


def _to16k(pcm24: bytes) -> np.ndarray:
    import librosa
    x = np.frombuffer(pcm24[: len(pcm24) // 2 * 2], "<i2").astype(np.float32) / 32768
    return librosa.resample(x, orig_sr=24000, target_sr=16000)


def stream(text: str, voice: str = VOICE, style: str = STYLE):
    """Yield float32 16 kHz chunks as they arrive. Raises if no backend works."""
    payload = {"t": text, "v": voice, "s": style, "fmt": 2}
    hit = G.cache_get("tts", payload, ext="pcm")
    if hit is not None:
        x = np.frombuffer(hit, "<f4")
        for k in range(0, len(x), 8000):
            yield x[k:k + 8000]
        return
    from google.genai import types
    cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    last, got = None, []
    for m in G.TTS_MODELS:
        try:
            buf = b""
            for ch in G.client().models.generate_content_stream(model=m, contents=prompt(text, style), config=cfg):
                for p in (ch.candidates[0].content.parts if ch.candidates and ch.candidates[0].content else []):
                    if p.inline_data and p.inline_data.data:
                        data = p.inline_data.data
                        if "wav" in (p.inline_data.mime_type or "") and data[:4] == b"RIFF":
                            data = data[44:]
                        buf += data
                        # emit in ~0.25 s pieces as they stream in
                        while len(buf) >= 12000:
                            y = _to16k(buf[:12000]); buf = buf[12000:]
                            got.append(y); yield y
            if buf:
                y = _to16k(buf); got.append(y); yield y
            if got:
                G.cache_put("tts", payload, np.concatenate(got).astype("<f4").tobytes(), ext="pcm")
                return
        except Exception as e:  # noqa: BLE001
            last = e
            if got:      # partial audio already delivered; stop here
                return
            continue
    raise last or RuntimeError("no TTS backend")


def synthesize(text: str, style: str | None = None) -> bytes | None:
    """Whole utterance as a WAV (for /api/tts)."""
    if not G.available():
        return None
    import soundfile as sf
    x = np.concatenate(list(stream(text, style=style or STYLE)))
    b = io.BytesIO(); sf.write(b, x, 16000, format="WAV", subtype="PCM_16")
    return b.getvalue()


def gemini_tts(text: str, voice="Kore", style: str | None = None) -> bytes:
    """WAV at 16 kHz (used by scripts/make_samples_gemini.py)."""
    import soundfile as sf
    x = np.concatenate(list(stream(text, voice=voice, style=style or STYLE)))
    b = io.BytesIO(); sf.write(b, x, 16000, format="WAV", subtype="PCM_16")
    return b.getvalue()
