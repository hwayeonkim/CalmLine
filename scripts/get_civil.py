import os, json, random
from datasets import load_dataset
ds = load_dataset("google/civil_comments", split="test")
random.seed(0)
pos = [i for i, t in enumerate(ds["toxicity"]) if t >= 0.5]
neg = [i for i, t in enumerate(ds["toxicity"]) if t < 0.1]
thr = [i for i, t in enumerate(ds["threat"]) if t >= 0.5]
sel = sorted(set(random.sample(pos, 1500) + random.sample(neg, 1500) + thr))
os.makedirs("data/external", exist_ok=True)
with open("data/external/civil_test_sample.jsonl", "w") as f:
    for i in sel:
        r = ds[i]; f.write(json.dumps({k: r[k] for k in ["text", "toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]}) + "\n")
print(len(ds), len(pos), len(thr), len(sel))
