import numpy as np
from calmline.textnorm import fact_present, leaked, numbers


def score_item(row, d, residual_tox=None):
    g = row["route"]
    toxic_gold = g in ("CALM", "ESCALATE")
    facts = row["facts"]
    fr = [fact_present(f, d.delivered) for f in facts]
    leaks = [s for s in row["toxic_spans"] if leaked(s, d.delivered)]
    halluc = sorted(numbers(d.delivered) - numbers(row["text"]) - {"311"})
    return dict(
        id=row["id"], gold=g, pred=d.route, route_ok=d.route in row["route_ok"],
        toxic_gold=toxic_gold, strict_pass=(g == "PASS" and row["route_ok"] == ["PASS"]),
        fact_recall=(sum(fr) / len(fr)) if fr else 1.0, sufficient=all(fr), missing=[f for f, ok in zip(facts, fr) if not ok],
        leaks=leaks, leaked=bool(leaks), halluc=halluc,
        intent_gold=row["intent"], intent_pred=d.intent, tier=d.tier, intent_score=d.intent_score,
        asserted=(d.route != "PASS" and d.intent != "UNKNOWN" and not d.delivered.split("] ")[-1].startswith(("Hello, I'm calling about a problem", "Hello, I'm calling to ask about an issue"))),
        latency=sum(d.timings_ms.values()), delivered=d.delivered, residual_tox=residual_tox,
    )


def aggregate(items):
    tox = [i for i in items if i["toxic_gold"]]
    thr = [i for i in items if i["gold"] == "ESCALATE"]
    cln = [i for i in items if i["strict_pass"]]
    pred_esc = [i for i in items if i["pred"] == "ESCALATE"]
    routed = [i for i in items if i["pred"] != "PASS"]
    lat = [i["latency"] for i in items]
    m = dict(
        n=len(items),
        route_acc=np.mean([i["route_ok"] for i in items]),
        abuse_recall=np.mean([i["pred"] != "PASS" for i in tox]),
        leak_rate=np.mean([i["leaked"] for i in tox]),
        protected=np.mean([(i["pred"] != "PASS") and not i["leaked"] for i in tox]),
        threat_recall=np.mean([i["pred"] == "ESCALATE" for i in thr]) if thr else float("nan"),
        threat_precision=np.mean([i["gold"] == "ESCALATE" for i in pred_esc]) if pred_esc else float("nan"),
        false_positive_rate=np.mean([i["pred"] != "PASS" for i in cln]) if cln else float("nan"),
        fact_recall_toxic=np.mean([i["fact_recall"] for i in tox]),
        sufficiency_toxic=np.mean([i["sufficient"] for i in tox]),
        fact_recall_all=np.mean([i["fact_recall"] for i in items]),
        halluc_rate=np.mean([bool(i["halluc"]) for i in items]),
        intent_acc_routed=np.mean([i["intent_pred"] == i["intent_gold"] for i in routed]) if routed else float("nan"),
        intent_acc_all=np.mean([i["intent_pred"] == i["intent_gold"] for i in items]),
        wrong_issue_rate=np.mean([i.get("asserted", False) and i["intent_pred"] != i["intent_gold"] for i in routed]) if routed else float("nan"),
        tier_dist={t: int(sum(i["tier"] == t for i in items)) for t in (0, 1, 2, 3)},
        latency_p50=float(np.percentile(lat, 50)), latency_p95=float(np.percentile(lat, 95)),
    )
    rt = [i["residual_tox"] for i in tox if i.get("residual_tox") is not None]
    if rt:
        m["residual_toxicity"] = np.mean([x >= 0.5 for x in rt])
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}
