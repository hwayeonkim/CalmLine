"""Utterance-level detector benchmark (abusive? threat?) on CalmLine dev sets and on Civil Comments.
Usage: python eval/detector_bench.py <dataset> [detectors...]"""
import json, os, sys, time
sys.path.insert(0, os.getcwd()); os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from calmline.toxicity import LexiconDetector, ModelDetector, HybridDetector, NLIDetector


def load(name):
    if name == "civil":
        rows = [json.loads(l) for l in open("data/external/civil_test_sample.jsonl")]
        return [(r["text"], r["toxicity"] >= 0.5, r["threat"] >= 0.5) for r in rows]
    rows = [json.loads(l) for l in open(f"data/gold/{name}.jsonl")]
    return [(r["text"], r["route"] != "PASS" and r["route_ok"] != ["PASS", "CALM"], r["route"] == "ESCALATE") for r in rows
            if r["route_ok"] != ["PASS", "CALM"]]


DET = {
    "lexicon_v0": lambda: LexiconDetector(extended=False),
    "lexicon_v4": lambda: LexiconDetector(extended="v4"),
    "toxic-bert": lambda: ModelDetector("unitary/toxic-bert"),
    "unbiased-roberta": lambda: ModelDetector("unitary/unbiased-toxic-roberta"),
    "hybrid_v4": lambda: HybridDetector(ModelDetector("unitary/toxic-bert"), LexiconDetector(extended="v4"), directed_gate=True),
    "nli": lambda: NLIDetector(),
    "hybrid_v4_tb": lambda: HybridDetector(ModelDetector("unitary/toxic-bert"), LexiconDetector(extended="v4"), directed_gate=True, threat_bypass=True),
    "hybrid_unb_tb": lambda: HybridDetector(ModelDetector("unitary/unbiased-toxic-roberta"), LexiconDetector(extended="v4"), directed_gate=True, threat_bypass=True),
}

if __name__ == "__main__":
    ds = sys.argv[1]
    names = sys.argv[2:] or list(DET)
    data = load(ds)
    res = {}
    for n in names:
        d = DET[n](); d(data[0][0])
        t0 = time.time(); outs = [d(t) for t, _, _ in data]; ms = (time.time() - t0) * 1000 / len(data)
        y, yt = np.array([a for _, a, _ in data]), np.array([b for _, _, b in data])
        p, pt = np.array([o.is_toxic for o in outs]), np.array([o.is_threat for o in outs])
        s, st = np.array([o.score for o in outs]), np.array([o.threat_score for o in outs])
        P, R, F, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
        TP, TR, TF, _ = precision_recall_fscore_support(yt, pt, average="binary", zero_division=0)
        r = dict(n=len(data), abuse_P=P, abuse_R=R, abuse_F1=F, abuse_AUC=roc_auc_score(y, s) if len(set(s)) > 2 else float("nan"),
                 threat_P=TP, threat_R=TR, threat_F1=TF, threat_AUC=roc_auc_score(yt, st) if len(set(st)) > 2 and yt.any() else float("nan"), ms=ms)
        res[n] = {k: round(float(v), 4) if not isinstance(v, int) else v for k, v in r.items()}
        print(ds, n, json.dumps(res[n]), flush=True)
    os.makedirs("results/detectors", exist_ok=True)
    out = f"results/detectors/{ds}.json"
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(res); json.dump(old, open(out, "w"), indent=1)
