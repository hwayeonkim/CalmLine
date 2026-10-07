"""Probe: per-part audio -> Gemini Live (manual activity, audio sent as fast as possible) -> calm speech.
Measures response latency per part and what the model heard/said."""
import asyncio, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np, soundfile as sf
from google.genai import types
from calmline import gemini as G
sys.path.insert(0, "experiments")
from live_s2s_probe import SYS

MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")


def parts_by_silence(x, sr=16000, min_sil=0.6):
    fr = np.array([np.abs(x[i:i + 320]).max() for i in range(0, len(x), 320)])
    voiced = fr > 0.02; out = []; st = None; sil = 0
    for i, v in enumerate(voiced):
        if v:
            if st is None: st = i
            sil = 0
        elif st is not None:
            sil += 1
            if sil * 0.02 >= min_sil: out.append((st * 320, (i - sil + 1) * 320)); st = None; sil = 0
    if st is not None: out.append((st * 320, len(x)))
    return out


async def main(path):
    x = sf.read(path)[0].astype(np.float32)
    cfg = types.LiveConnectConfig(response_modalities=["AUDIO"], system_instruction=SYS,
        input_audio_transcription=types.AudioTranscriptionConfig(), output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))),
        realtime_input_config=types.RealtimeInputConfig(automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)))
    async with G.client().aio.live.connect(model=MODEL, config=cfg) as s:
        for a, b in parts_by_silence(x):
            seg = (np.clip(x[a:b], -1, 1) * 32767).astype("<i2").tobytes()
            t0 = time.time()
            await s.send_realtime_input(activity_start=types.ActivityStart())
            for k in range(0, len(seg), 32000):
                await s.send_realtime_input(audio=types.Blob(data=seg[k:k + 32000], mime_type="audio/pcm;rate=16000"))
            await s.send_realtime_input(activity_end=types.ActivityEnd())
            t_sent = time.time() - t0
            first = None; heard = ""; said = ""; n = 0
            async for m in s.receive():
                sc = m.server_content
                if sc is None: continue
                if sc.input_transcription and sc.input_transcription.text: heard += sc.input_transcription.text
                if sc.output_transcription and sc.output_transcription.text: said += sc.output_transcription.text
                if sc.model_turn:
                    for p in sc.model_turn.parts:
                        if p.inline_data and p.inline_data.data:
                            first = first or (time.time() - t0); n += len(p.inline_data.data)
                if sc.turn_complete: break
            print(f"part {(b-a)/16000:4.1f}s | send {t_sent:.2f}s | first audio {first if first is None else round(first,2)}s | out {n/48000:.1f}s")
            print(f"     heard: {heard.strip()[:110]}\n     said : {said.strip()[:110]}")

asyncio.run(main(sys.argv[1]))
