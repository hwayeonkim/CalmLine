"""Google text-to-speech for the 'Calm voice' mode (and for regenerating demo calls).

Backends, in order: Gemini TTS (same GEMINI_API_KEY as the rewriter) -> Google Cloud Text-to-Speech
(ADC credentials) -> none (the agent page falls back to the browser's speechSynthesis).
NOTE: written without credentials available; verify the model id in AI Studio before relying on it."""
from __future__ import annotations
import io
import os

import numpy as np

GEMINI_TTS_MODEL = os.getenv("CALMLINE_TTS_MODEL", "gemini-2.5-flash-preview-tts")


def _wav(pcm16: bytes, sr: int) -> bytes:
    import soundfile as sf
    x = np.frombuffer(pcm16, "<i2").astype(np.float32) / 32768
    b = io.BytesIO(); sf.write(b, x, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def gemini_tts(text: str, voice="Kore", style: str | None = None) -> bytes:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"{style}: {text}" if style else text
    r = client.models.generate_content(
        model=GEMINI_TTS_MODEL, contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
    pcm = r.candidates[0].content.parts[0].inline_data.data   # 24 kHz mono s16le
    return _wav(pcm, 24000)


def cloud_tts(text: str, voice="en-US-Neural2-F") -> bytes:
    from google.cloud import texttospeech as tts
    c = tts.TextToSpeechClient()
    r = c.synthesize_speech(input=tts.SynthesisInput(text=text), voice=tts.VoiceSelectionParams(language_code="en-US", name=voice),
                            audio_config=tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=24000))
    return r.audio_content


def available() -> str | None:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    try:
        import google.cloud.texttospeech  # noqa: F401
        import google.auth
        google.auth.default()
        return "cloud"
    except Exception:
        return None


def synthesize(text: str, style: str | None = None) -> bytes | None:
    b = available()
    if b == "gemini":
        return gemini_tts(text, style=style or "Say calmly and politely")
    if b == "cloud":
        return cloud_tts(text)
    return None
