"""S1 spike from the plan: does Gemini accept abusive caller text (content-moderation use) and rewrite it
faithfully? Run once you have a key:

    GEMINI_API_KEY=... .venv/bin/python scripts/gemini_s1_spike.py [model]

Reports: blocked / empty responses, verification pass rate, latency p50/p95, on the DEV set only."""
import json, os, sys, time
sys.path.insert(0, os.getcwd()); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from calmline.llm import GeminiRewriter, VerifiedRewriter
from calmline.versions import build

model = sys.argv[1] if len(sys.argv) > 1 else None
g = GeminiRewriter(model)
p = build("v10")
v = VerifiedRewriter(g, p.detector, p.renderer)
rows = [json.loads(l) for l in open("data/gold/dev.jsonl") if json.loads(l)["route"] != "PASS"]
blocked, lat = 0, []
for r in rows:
    tox = p.detector(r["text"])
    t0 = time.time()
    try:
        out = g.generate(r["text"])
    except Exception as e:
        out = ""; print("ERROR", type(e).__name__, str(e)[:120])
    lat.append(time.time() - t0)
    if not out: blocked += 1
    reason = v.check(r["text"], out, __import__("calmline.slots", fromlist=["extract"]).extract(r["text"])) if out else "empty"
    print(f"[{reason or 'OK'}] {r['text'][:70]!r}\n      -> {out[:160]!r}")
print(json.dumps(dict(model=g.model, n=len(rows), blocked_or_empty=blocked, latency_p50=float(np.percentile(lat, 50)),
                      latency_p95=float(np.percentile(lat, 95))), indent=1))
