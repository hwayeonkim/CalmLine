"""Threshold sweep for the v8 detector on DEV data only (dev + synth + hardneg)."""
import json, os, sys, itertools
sys.path.insert(0, os.getcwd()); os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from calmline.toxicity import HybridDetector, ModelDetector, LexiconDetector
rows = []
for s in ["dev", "synth", "hardneg"]:
    rows += [json.loads(l) for l in open(f"data/gold/{s}.jsonl")]
rows = [r for r in rows if r["route_ok"] != ["PASS", "CALM"]]
res = []
for tox_th, conf, thr_th in itertools.product([0.5, 0.6, 0.7, 0.8], [0.85, 0.9, 0.95, 0.99], [0.3, 0.5, 0.7]):
    m = ModelDetector("unitary/unbiased-toxic-roberta", tox_th=tox_th, threat_th=thr_th)
    d = HybridDetector(m, LexiconDetector(extended="v7"), directed_gate=True, threat_bypass=True, context_filter=True, confident=conf)
    out = [d(r["text"]) for r in rows]
    ab = [o.is_toxic for o, r in zip(out, rows) if r["route"] != "PASS"]
    cl = [o.is_toxic for o, r in zip(out, rows) if r["route"] == "PASS"]
    th = [o.is_threat for o, r in zip(out, rows) if r["route"] == "ESCALATE"]
    thp = [r["route"] == "ESCALATE" for o, r in zip(out, rows) if o.is_threat]
    res.append(dict(tox_th=tox_th, conf=conf, threat_th=thr_th, abuse_recall=np.mean(ab), fpr=np.mean(cl), threat_recall=np.mean(th), threat_prec=np.mean(thp)))
    print(res[-1], flush=True)
json.dump(res, open("results/detectors/sweep_v8.json", "w"), indent=1, default=float)
