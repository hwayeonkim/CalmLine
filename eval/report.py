"""Render results/RESULTS.md from all result JSON files."""
import sys; sys.path.insert(0, ".")
import json, os, glob
S = json.load(open("results/summary.json"))
from calmline.versions import DESCRIPTIONS
order = sorted([v for v in S if v.startswith("v")], key=lambda v: (int(''.join(c for c in v[1:] if c.isdigit()) or 0), v))
pct = lambda x: "—" if x is None or x != x else f"{100*x:.1f}"
L = ["# CalmLine — experiment results", "",
     "All numbers are produced by `python -m eval.run_text <versions>`; every version is a frozen config in `calmline/versions.py`.",
     "", "## Text pipeline — held-out hand-written TEST set (n=134, frozen before any tuning)", "",
     "| ver | protected ↑ | leak ↓ | abuse recall ↑ | threat recall ↑ | threat prec ↑ | FPR test ↓ | FPR hard-neg ↓ | sufficiency ↑ | fact recall ↑ | halluc ↓ | intent acc ↑ | p95 ms |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for v in order:
    t = S[v].get("test"); h = S[v].get("hardneg", {})
    if not t: continue
    L.append(f"| {v} | {pct(t['protected'])} | {pct(t['leak_rate'])} | {pct(t['abuse_recall'])} | {pct(t['threat_recall'])} | {pct(t['threat_precision'])} | "
             f"{pct(t['false_positive_rate'])} | {pct(h.get('false_positive_rate'))} | {pct(t['sufficiency_toxic'])} | {pct(t['fact_recall_toxic'])} | "
             f"{pct(t['halluc_rate'])} | {pct(t['intent_acc_routed'])} | {t['latency_p95']:.0f} |")
L += ["", "*protected* = abusive utterance detected **and** no abusive span reached the agent. *sufficiency* = every gold fact needed to resolve the complaint survives in what the agent receives (abusive utterances only). *FPR* = clean utterances that were rewritten. *halluc* = numbers in the agent's text that the caller never said.", ""]
L += ["## Same metrics on development data (used for tuning)", "", "| ver | split | protected | leak | threat recall | FPR | sufficiency |", "|---|---|---|---|---|---|---|"]
for v in order:
    for sp in ("dev", "synth"):
        t = S[v].get(sp)
        if t: L.append(f"| {v} | {sp} | {pct(t['protected'])} | {pct(t['leak_rate'])} | {pct(t['threat_recall'])} | {pct(t['false_positive_rate'])} | {pct(t['sufficiency_toxic'])} |")
L += ["", "## Version log", ""] + [f"- **{v}** — {DESCRIPTIONS.get(v, '')}" for v in order]
if os.path.exists("results/detectors/bench_log.txt"):
    L += ["", "## Detector benchmark (utterance level)", "", "| set | detector | abuse P | abuse R | abuse F1 | threat R | threat AUC | ms/utt |", "|---|---|---|---|---|---|---|---|"]
    for line in open("results/detectors/bench_log.txt"):
        ds, name, js = line.split(" ", 2); d = json.loads(js.replace("NaN", "null"))
        f = lambda k: "—" if d.get(k) is None else f"{d[k]:.3f}"
        L.append(f"| {ds} | {name} | {f('abuse_P')} | {f('abuse_R')} | {f('abuse_F1')} | {f('threat_R')} | {f('threat_AUC')} | {d['ms']:.1f} |")
if os.path.exists("results/audio/acoustic_cremad.json"):
    a = json.load(open("results/audio/acoustic_cremad.json"))
    L += ["", f"## Acoustic aggression (CREMA-D, speaker-independent: {a['n_test_speakers']} held-out speakers, n={a['n_test']})", "",
          "| features / model | AUC | recall | recall (medium+high anger) | precision |", "|---|---|---|---|---|"]
    for k, v in a.items():
        if isinstance(v, dict) and "auc" in v:
            L.append(f"| {k} | {v['auc']:.3f} | {v['recall']:.3f} | {v['recall_anger_HI_MD']:.3f} | {v['precision']:.3f} |")
    c = a["compressor_peak_reduction_db"]
    L.append(f"\nLook-ahead compressor: peak loudness reduced by **{c['ANG_HI']['mean']:.1f} dB** on high-intensity anger vs **{c['NEU']['mean']:.1f} dB** on neutral speech.")
streams = sorted(glob.glob("results/audio/stream_*.json"))
if streams:
    L += ["", "## Real-time delay buffer — abusive-word leak vs. playback delay (TEST audio, TTS-rendered)", "",
          "| STT | chunk | policy | RTF | final-card latency p50/p95 (s) | leak@0.5s | leak@1s | leak@1.5s | leak@2s | leak@3s | collateral s/clean utt @2s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in streams:
        s = json.load(open(p))["summary"]
        L.append(f"| {s['whisper']} | {s['chunk_s']} | {s['policy']} | {s['rtf_mean']:.2f} | {s['final_latency_p50']:.2f} / {s['final_latency_p95']:.2f} | "
                 f"{pct(s['word_leak@0.5'])} | {pct(s['word_leak@1.0'])} | {pct(s['word_leak@1.5'])} | {pct(s['word_leak@2.0'])} | {pct(s['word_leak@3.0'])} | {s['collateral_s_per_clean_utt@2.0']:.2f} |")
open("results/RESULTS.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[:30]))
