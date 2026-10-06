"""Demo sample calls (fictional, acted with macOS TTS). [[...]] are `say` prosody commands used to
make the agitated parts louder/faster/higher so the acoustic gauge has something to react to."""
import json, os, re, subprocess
OUT = "app/static/samples"
LOUD = "[[volm 1.0]][[rate 225]][[pbas +12]]"
CALM = "[[volm 0.55]][[rate 185]][[pbas +0]]"
S = [
 dict(id="heat", title="No heat — profanity, three kids", voice="Samantha",
      say=f"{CALM}Hi. I'm calling about the heat again. {LOUD}This is the fourth fucking time I've called! {CALM}We've had no heat at 1820 Walton Avenue, apartment 2C, since Wednesday. {LOUD}I have three kids freezing in here, so get your lazy asses moving!"),
 dict(id="pothole", title="Pothole — insults aimed at the agent", voice="Daniel",
      say=f"{LOUD}Are you people stupid or something? {CALM}There's a pothole on Linden Boulevard near Euclid Avenue, it blew out my tire yesterday. {LOUD}Write it down, genius, or is that too hard for you? {CALM}I want someone to call me back."),
 dict(id="harass", title="Noise complaint — sexual harassment of the agent", voice="Rishi",
      say=f"{CALM}Hey sweetheart, you sound cute. What are you wearing right now? {CALM}Anyway, my upstairs neighbor in apartment 9C at 75 Hart Street plays drums every night after midnight."),
 dict(id="threat", title="Threat — escalates to supervisor", voice="Karen",
      say=f"{LOUD}I've called six times about the parked truck on my hydrant on Court Street! {LOUD}I know where your office is. If nobody tows it today I'm going to come down there and beat the crap out of somebody."),
 dict(id="civil", title="Frustrated but civil — passes through untouched", voice="Moira",
      say=f"{CALM}I'm really frustrated, this is the third time I'm calling. Water is still leaking through my bathroom ceiling at 222 East 21st Street, apartment 4B. My reference number is 311-20871455. Could you send an inspector this week?"),
 dict(id="identity", title="Identity attack + open hydrant", voice="Tessa",
      say=f"{LOUD}Put someone on who speaks real English. {CALM}A hydrant is open on Kingston Avenue at Eastern Parkway and the street is flooding. {LOUD}You people are useless."),
]
os.makedirs(OUT, exist_ok=True)
meta = []
for s in S:
    wav = f"{OUT}/{s['id']}.wav"
    subprocess.run(["say", "-v", s["voice"], "-o", wav, "--data-format=LEI16@16000", s["say"]], check=True)
    meta.append(dict(id=s["id"], title=s["title"], text=re.sub(r"\[\[.*?\]\]", "", s["say"])))
json.dump(meta, open(f"{OUT}/samples.json", "w"), indent=1)
print(len(meta))
