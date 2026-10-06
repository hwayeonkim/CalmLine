import json, sys
v, split = sys.argv[1], sys.argv[2]
what = sys.argv[3] if len(sys.argv) > 3 else "all"
items = json.load(open(f"results/{v}_{split}_items.json"))
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/gold/{split}.jsonl")}
for i in items:
    bad = (not i["route_ok"]) or i["leaked"] or (i["toxic_gold"] and not i["sufficient"]) or i["halluc"]
    if what == "route" and i["route_ok"]: continue
    if what == "leak" and not i["leaked"]: continue
    if what == "facts" and (i["sufficient"] or not i["toxic_gold"]): continue
    if what == "all" and not bad: continue
    print(f"[{i['id']}] gold={i['gold']} pred={i['pred']} tier={i['tier']} intent={i['intent_pred']}/{i['intent_gold']}")
    print("   IN :", rows[i["id"]]["text"])
    print("   OUT:", i["delivered"])
    if i["leaks"]: print("   LEAK:", i["leaks"])
    if i["missing"]: print("   MISSING:", i["missing"])
