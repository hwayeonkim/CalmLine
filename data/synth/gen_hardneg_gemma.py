"""Hard negatives written by a *different author* (Gemma 3 1B): civil callers who use violent / scary /
figurative vocabulary without abusing anyone. These must PASS. Used to tune false-positive behaviour."""
import json, os, re, sys, random
sys.path.insert(0, os.getcwd()); os.environ["HF_HUB_OFFLINE"] = "1"
from calmline.llm import _load_gemma
tok, mdl = _load_gemma("unsloth/gemma-3-1b-it", "mps")
PROMPTS = [
 "Write 6 different things a polite but scared New York resident might say when calling 311 to report a problem (street light out, broken door lock, encampment, dark park) where they mention crime or violence they witnessed or fear, e.g. a shooting or mugging nearby. They are NOT threatening anyone. Natural spoken English, one per line, no numbering.",
 "Write 6 different things a frustrated New York resident might say to 311 using dramatic figurative expressions like 'this is killing me', 'I'm dying here', 'it's murder', 'I could scream', while reporting a real problem like noise, heat, rats or a leak. They do not insult the operator. One per line, no numbering.",
 "Write 6 different things a worried caller might tell 311 about a dangerous situation: a tree about to fall, a gas smell, an exposed live wire, a hole in the road. Use urgent language like 'someone is going to die' or 'this will kill somebody'. They are polite to the operator. One per line, no numbering.",
 "Write 6 different things a New York tenant might say to 311 about a neighbor who is threatening or harassing THEM (the caller), asking for help. The caller is polite to the operator. One per line, no numbering.",
]
import torch
out = []
random.seed(1); torch.manual_seed(1)
for rep in range(4):
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("mps")
        g = mdl.generate(**ids, max_new_tokens=320, do_sample=True, temperature=1.0, top_p=0.95)
        txt = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        for line in txt.split("\n"):
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('“”"')
            if 25 < len(line) < 300 and not line.lower().startswith(("here are", "okay", "sure", "note")):
                out.append(line)
seen, rows = set(), []
for t in out:
    k = t.lower()[:60]
    if k in seen: continue
    seen.add(k)
    rows.append(dict(id=f"hn-{len(rows):03d}", text=t, route="PASS", route_ok=["PASS"], intent="UNKNOWN", facts=[], toxic_spans=[], source="gemma-hardneg"))
with open("data/gold/hardneg.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows))
