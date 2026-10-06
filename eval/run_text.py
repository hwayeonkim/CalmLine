"""Run versions over gold sets: python -m eval.run_text v0 v1 ... [--splits dev,test]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from calmline.versions import build, DESCRIPTIONS
from calmline.toxicity import ModelDetector
from eval.metrics import score_item, aggregate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(split):
    return [json.loads(l) for l in open(os.path.join(ROOT, "data", "gold", f"{split}.jsonl"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="+")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    judge = None if a.no_judge else ModelDetector("unitary/unbiased-toxic-roberta", per_clause=False)
    summary_path = os.path.join(ROOT, "results", "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for v in a.versions:
        pipe = build(v)
        for split in a.splits.split(","):
            rows = load(split)
            pipe(rows[0]["text"])  # warm-up (model load) excluded from latency
            items = []
            for r in rows:
                d = pipe(r["text"])
                rt = None
                if judge is not None and r["route"] != "PASS":
                    rt = judge.tox_of(judge.scores([d.delivered])[0]) if d.delivered.strip() else 0.0
                items.append(score_item(r, d, rt))
            m = aggregate(items)
            if hasattr(pipe.llm, "stats"):
                st = dict(pipe.llm.stats); ms = st.pop("ms")
                m["llm"] = dict(st, ms_p50=float(__import__("numpy").percentile(ms, 50)) if ms else None,
                                ms_p95=float(__import__("numpy").percentile(ms, 95)) if ms else None)
            summary.setdefault(v, {})[split] = m
            summary[v]["desc"] = DESCRIPTIONS.get(v, "")
            json.dump(items, open(os.path.join(ROOT, "results", f"{v}_{split}_items.json"), "w"), indent=1)
            print(v, split, json.dumps(m))
    json.dump(summary, open(summary_path, "w"), indent=1)


if __name__ == "__main__":
    main()
