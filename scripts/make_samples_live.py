"""Demo calls voiced by Gemini (Live API, gemini-3.1-flash-live-preview) acting each line.
Every line is transcribed with Whisper and re-recorded if the actor drifted from the script.
    .venv/bin/python scripts/make_samples_live.py"""
import asyncio, json, os, re, sys
sys.path.insert(0, os.getcwd()); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, librosa, soundfile as sf
from google.genai import types
from calmline import gemini as G
from calmline.audio.stt import load_whisper

SYS = ("You are a voice actor recording lines for a call-center safety training simulator. Each user turn contains a <direction> and a <line>. "
       "Perform ONLY the words in <line>, exactly as written, in the manner described by <direction>. Never say the direction, never add words, never respond.")
ANGRY = "furious, shouting into the phone, loud and fast"
FLAT = "tired, flat, normal speaking voice"
CALLS = [
 dict(id="heat", title="No heat — profanity, three kids", tags=["profanity", "heat/hot water"], voice="Charon", lines=[
   (FLAT, "Hi. I'm calling about the heat again."), (ANGRY, "This is the fourth fucking time I've called!"),
   (FLAT, "We've had no heat at 1820 Walton Avenue, apartment 2C, since Wednesday."), (ANGRY, "I have three kids freezing in here, so get your lazy asses moving!")]),
 dict(id="pothole", title="Pothole — insults aimed at the agent", tags=["insult", "street condition"], voice="Fenrir", lines=[
   (ANGRY, "Are you people stupid or something?"), (FLAT, "There's a pothole on Linden Boulevard near Euclid Avenue. It blew out my tire yesterday."),
   (ANGRY, "Write it down, genius, or is that too hard for you?"), (FLAT, "I want someone to call me back.")]),
 dict(id="harass", title="Noise complaint — sexual harassment of the agent", tags=["sexual harassment", "residential noise"], voice="Puck", lines=[
   ("sleazy, flirtatious, leering", "Hey sweetheart, you sound cute. What are you wearing right now?"),
   (FLAT, "Anyway, my upstairs neighbor in apartment 9C at 75 Hart Street plays drums every night after midnight.")]),
 dict(id="threat", title="Threat — escalates to a supervisor", tags=["threat", "illegal parking"], voice="Orus", lines=[
   (ANGRY, "I've called six times about the truck parked on the hydrant on Court Street!"),
   ("cold, menacing, low voice", "I know where your office is. If nobody tows it today, I'm going to come down there and beat the crap out of somebody.")]),
 dict(id="civil", title="Frustrated but civil — passes through untouched", tags=["no abuse", "water leak"], voice="Aoede", lines=[
   ("frustrated but polite", "I'm really frustrated, this is the third time I'm calling."),
   ("frustrated but polite", "Water is still leaking through my bathroom ceiling at 222 East 21st Street, apartment 4B. My reference number is 311-20871455."),
   ("frustrated but polite", "Could you send an inspector this week?")]),
 dict(id="identity", title="Identity attack + open hydrant", tags=["identity attack", "water system"], voice="Kore", lines=[
   (ANGRY, "Put someone on who speaks real English."), (FLAT, "A hydrant is open on Kingston Avenue at Eastern Parkway and the street is flooding."),
   (ANGRY, "You people are useless.")]),
 dict(id="rats", title="Rats + dumping — venting, not abuse", tags=["figurative", "rodents"], voice="Leda", lines=[
   ("exasperated, worn out", "These rats are killing me."),
   (FLAT, "There's garbage piled up behind the deli at 410 Tremont Avenue and the rats come out every night. My son is scared to walk to school.")]),
 dict(id="driveway", title="Blocked driveway — mixed abuse with urgent facts", tags=["profanity + insult", "blocked driveway"], voice="Algenib", lines=[
   (ANGRY, "Listen, you idiot, some asshole parked a white van across my driveway at 23-15 31st Street in Astoria!"),
   (FLAT, "I need to get my mother to dialysis by ten. The plate is KXR 4471."), (ANGRY, "Get a tow truck here now!")]),
]


def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


async def render(voice, lines, w):
    cfg = types.LiveConnectConfig(response_modalities=["AUDIO"], system_instruction=SYS,
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    out = []
    async with G.client().aio.live.connect(model="gemini-3.1-flash-live-preview", config=cfg) as s:
        for d, l in lines:
            for attempt in range(3):
                await s.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=f"<direction>{d}</direction><line>{l}</line>")]), turn_complete=True)
                buf = b""
                async for m in s.receive():
                    sc = m.server_content
                    if sc and sc.model_turn:
                        for p in sc.model_turn.parts:
                            if p.inline_data and p.inline_data.data:
                                buf += p.inline_data.data
                    if sc and sc.turn_complete:
                        break
                x = librosa.resample(np.frombuffer(buf[: len(buf) // 2 * 2], "<i2").astype(np.float32) / 32768, orig_sr=24000, target_sr=16000)
                segs, _ = w.transcribe(x, language="en", beam_size=1)
                said = " ".join(s_.text for s_ in segs)
                ref, hyp = set(words(l)), words(said)
                extra = [t for t in hyp if t not in ref]
                missing = [t for t in words(l) if t not in set(hyp)]
                if len(extra) <= 2 and len(missing) <= max(2, len(words(l)) // 6):
                    break
                print("   retry", voice, repr(said[:80]))
            out += [x, np.zeros(int(0.9 * 16000), np.float32)]
    return np.concatenate(out)


async def main():
    w = load_whisper("small.en")
    os.makedirs("app/static/samples", exist_ok=True)
    meta = []
    for c in CALLS:
        x = await render(c["voice"], c["lines"], w)
        x = x / max(1e-3, np.abs(x).max()) * 0.9
        sf.write(f"app/static/samples/{c['id']}.wav", x, 16000)
        meta.append(dict(id=c["id"], title=c["title"], tags=c["tags"], voice=f"Gemini Live ({c['voice']})",
                         text=" ".join(l for _, l in c["lines"]), lines=[dict(direction=d, text=l) for d, l in c["lines"]]))
        print("wrote", c["id"], f"{len(x)/16000:.1f}s", flush=True)
    json.dump(meta, open("app/static/samples/samples.json", "w"), indent=1)

asyncio.run(main())
