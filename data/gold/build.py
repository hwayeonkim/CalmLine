import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
RMAP = {"P": "PASS", "C": "CALM", "E": "ESCALATE", "M": "PASS"}
for name in ["dev", "test"]:
    mod = __import__(f"{name}_src")
    with open(os.path.join(os.path.dirname(__file__), f"{name}.jsonl"), "w") as f:
        for i, (text, r, intent, facts, toxic) in enumerate(mod.T):
            row = dict(id=f"{name}-{i:03d}", text=text, route=RMAP[r], route_ok=["PASS", "CALM"] if r == "M" else [RMAP[r]],
                       intent=intent, facts=facts, toxic_spans=toxic, source="handwritten")
            f.write(json.dumps(row) + "\n")
    print(name, len(mod.T))
