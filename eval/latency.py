"""Clean per-utterance latency (no score memo, warm models) on the test set."""
import json, os, sys, time
sys.path.insert(0, os.getcwd()); os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["CALMLINE_NO_MEMO"] = "1"
import numpy as np
from calmline.versions import build
rows = [json.loads(l) for l in open("data/gold/test.jsonl")]
out = {}
for v in sys.argv[1:]:
    p = build(v); [p(r["text"]) for r in rows[:5]]
    ms = []
    for r in rows:
        t0 = time.perf_counter(); p(r["text"]); ms.append((time.perf_counter() - t0) * 1000)
    out[v] = dict(p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)), max=float(max(ms)))
    print(v, {k: round(x, 1) for k, x in out[v].items()}, flush=True)
json.dump(out, open("results/latency_test.json", "w"), indent=1)
