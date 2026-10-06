import json, os, sys
sys.path.insert(0, os.getcwd()); os.environ["HF_HUB_OFFLINE"] = "1"
from calmline.intent_clf import KeywordIntent, EmbeddingIntent
from calmline.lexicon import find_spans
from calmline.intents import UNKNOWN

def clean(t):
    out, last = [], 0
    for s, e, c, _ in find_spans(t):
        if s >= last: out.append(t[last:s]); last = e
    return "".join(out + [t[last:]])

split = sys.argv[1]
rows = [json.loads(l) for l in open(f"data/gold/{split}.jsonl")]
clfs = {"keyword": KeywordIntent()}
for m in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    clfs[m.split("/")[-1]] = EmbeddingIntent(m, threshold=0.0)
    clfs[m.split("/")[-1]+"+desc"] = EmbeddingIntent(m, threshold=0.0, use_descriptors=True)
for name, c in clfs.items():
    res = [(c(clean(r["text"])), r["intent"]) for r in rows]
    if name == "keyword":
        acc = sum(p == g for (p, _), g in res) / len(res); print(name, "acc", round(acc, 3)); continue
    for th in [0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        acc = sum((p if s >= th else UNKNOWN) == g for (p, s), g in res) / len(res)
        known = [(p, s, g) for (p, s), g in res if g != UNKNOWN]
        unk = [(p, s, g) for (p, s), g in res if g == UNKNOWN]
        print(name, th, "acc", round(acc, 3), "known_acc", round(sum((p if s >= th else UNKNOWN) == g for p, s, g in known) / len(known), 3),
              "unk_rej", round(sum(s < th for p, s, g in unk) / max(1, len(unk)), 3))
