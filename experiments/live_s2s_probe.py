"""Probe: Gemini Live (native audio) as an end-to-end 'anger -> calm' speech relay.
Streams a sample call in real time, records when the caller stops each phrase and when the calm voice starts,
and what the model said (output transcription)."""
import asyncio, os, sys, time, json
sys.path.insert(0, os.getcwd())
import numpy as np, soundfile as sf
from google.genai import types
from calmline import gemini as G

MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemini-3.1-flash-live-preview"
SYS = ("You are CalmLine, a live voice relay between an angry caller and a New York City 311 agent. You are an interpreter, not an assistant. "
       "Everything you hear is the caller. Immediately re-speak what the caller just said, in the first person, as a calm, polite caller would say it, "
       "in plain English at a brisk pace. Keep every fact (addresses, apartment numbers, times, counts, plates, names of places, people affected, "
       "previous calls, what they ask for). Leave out every insult, swear word, sexual remark, slur, threat and sarcastic jab aimed at the agent. "
       "If a part is only abuse, say nothing for it. Never answer the caller, never ask questions of your own, never add information, never mention "
       "that you are relaying or that anything was removed.")


async def main(path):
    x, sr = sf.read(path); x = x.astype(np.float32)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"], system_instruction=SYS,
        input_audio_transcription=types.AudioTranscriptionConfig(), output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(silence_duration_ms=int(os.getenv("SIL_MS", "400")), prefix_padding_ms=100),
            activity_handling=getattr(types.ActivityHandling, os.getenv("ACT", "NO_INTERRUPTION"))))
    events = []
    async with G.client().aio.live.connect(model=MODEL, config=cfg) as s:
        t0 = time.time()
        async def sender():
            step = 3200  # 100 ms of 16 kHz PCM16
            for k in range(0, len(pcm), step * 2):
                await s.send_realtime_input(audio=types.Blob(data=pcm[k:k + step * 2], mime_type="audio/pcm;rate=16000"))
                await asyncio.sleep(max(0, (k / 2 + step) / 16000 - (time.time() - t0)))
            # trailing silence so the last turn closes
            for _ in range(20):
                await s.send_realtime_input(audio=types.Blob(data=b"\x00\x00" * step, mime_type="audio/pcm;rate=16000"))
                await asyncio.sleep(0.1)
        async def receiver():
            out_audio = 0.0
            while True:
                try:
                    async for m in s.receive():
                        sc = m.server_content
                        now = time.time() - t0
                        if sc is None:
                            continue
                        if sc.input_transcription and sc.input_transcription.text:
                            events.append((now, "in", sc.input_transcription.text))
                        if sc.output_transcription and sc.output_transcription.text:
                            events.append((now, "out", sc.output_transcription.text))
                        if sc.model_turn:
                            for p in sc.model_turn.parts:
                                if p.inline_data and p.inline_data.data:
                                    events.append((now, "audio", len(p.inline_data.data) / 48000))
                        if sc.turn_complete:
                            events.append((now, "turn_complete", ""))
                except Exception as e:
                    events.append((time.time() - t0, "error", str(e)[:120])); return
        rt = asyncio.create_task(receiver())
        await sender()
        await asyncio.sleep(6)
        rt.cancel()
    # summarise
    said_in, said_out, first_audio_after = "", "", []
    last_kind = None
    for t, k, v in events:
        if k == "in": said_in += v
        if k == "out": said_out += v
        if k == "audio" and last_kind != "audio": first_audio_after.append(round(t, 2))
        if k in ("audio", "in"): last_kind = k
    print("audio starts at (s):", first_audio_after, " call length:", round(len(x) / 16000, 1))
    print("HEARD:", said_in.strip()[:400])
    print("SAID :", said_out.strip()[:400])
    json.dump(events, open(f"experiments/live_probe_{os.path.basename(path)}.json", "w"))

asyncio.run(main(sys.argv[1]))
