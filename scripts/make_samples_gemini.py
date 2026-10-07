"""Regenerate the demo calls with Gemini TTS so the caller actually *sounds* angry (replaces macOS `say`).
    GEMINI_API_KEY=... .venv/bin/python scripts/make_samples_gemini.py
Each call is rendered segment by segment with a style instruction, then concatenated at 16 kHz."""
import io, json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, soundfile as sf, librosa
from calmline.tts import gemini_tts

ANGRY, CALM = "furious caller shouting into the phone, loud and fast", "tired caller, normal speaking voice"
CALLS = {
 "heat": ("Charon", [(CALM, "Hi. I'm calling about the heat again."), (ANGRY, "This is the fourth fucking time I've called!"),
                     (CALM, "We've had no heat at 1820 Walton Avenue, apartment 2C, since Wednesday."), (ANGRY, "I have three kids freezing in here, so get your lazy asses moving!")]),
 "pothole": ("Fenrir", [(ANGRY, "Are you people stupid or something?"), (CALM, "There's a pothole on Linden Boulevard near Euclid Avenue, it blew out my tire yesterday."),
                        (ANGRY, "Write it down, genius, or is that too hard for you?"), (CALM, "I want someone to call me back.")]),
 "harass": ("Puck", [("sleazy, flirtatious, leering", "Hey sweetheart, you sound cute. What are you wearing right now?"),
                     (CALM, "Anyway, my upstairs neighbor in apartment 9C at 75 Hart Street plays drums every night after midnight.")]),
 "threat": ("Orus", [(ANGRY, "I've called six times about the parked truck on my hydrant on Court Street!"),
                     ("cold, menacing, low voice", "I know where your office is. If nobody tows it today I'm going to come down there and beat the crap out of somebody.")]),
 "civil": ("Aoede", [("frustrated but polite", "I'm really frustrated, this is the third time I'm calling. Water is still leaking through my bathroom ceiling at 222 East 21st Street, apartment 4B. My reference number is 311-20871455. Could you send an inspector this week?")]),
 "identity": ("Kore", [(ANGRY, "Put someone on who speaks real English."), (CALM, "A hydrant is open on Kingston Avenue at Eastern Parkway and the street is flooding."), (ANGRY, "You people are useless.")]),
}
OUT = "app/static/samples"
meta = json.load(open(f"{OUT}/samples.json"))
for sid, (voice, segs) in CALLS.items():
    parts = []
    for style, text in segs:
        x, sr = sf.read(io.BytesIO(gemini_tts(text, voice=voice, style=style)))
        parts += [librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000), np.zeros(14400, np.float32)]
    sf.write(f"{OUT}/{sid}.wav", np.concatenate(parts), 16000)
    for m in meta:
        if m["id"] == sid: m["voice"] = f"Gemini TTS ({voice})"
    print("wrote", sid)
json.dump(meta, open(f"{OUT}/samples.json", "w"), indent=1)
