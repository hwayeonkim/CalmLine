"""Render gold utterances to 16 kHz mono WAV with macOS `say` (accent-diverse voices).
NOTE: TTS has no real anger prosody — used only for STT / streaming-latency / leak experiments,
never for acoustic-aggression evaluation (that uses CREMA-D human actors)."""
import json, os, subprocess, sys
VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa"]
for split in sys.argv[1:]:
    os.makedirs(f"data/audio/{split}", exist_ok=True)
    meta = []
    for i, l in enumerate(open(f"data/gold/{split}.jsonl")):
        r = json.loads(l); v = VOICES[i % len(VOICES)]
        out = f"data/audio/{split}/{r['id']}.wav"
        if not os.path.exists(out):
            subprocess.run(["say", "-v", v, "-r", "190", "-o", out, "--data-format=LEI16@16000", r["text"]], check=True)
        meta.append(dict(id=r["id"], voice=v, path=out))
    json.dump(meta, open(f"data/audio/{split}/meta.json", "w"), indent=1)
    print(split, len(meta))
