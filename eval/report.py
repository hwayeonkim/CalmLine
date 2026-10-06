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
     "| ver | protected ↑ | leak ↓ | abuse recall ↑ | threat recall ↑ | threat prec ↑ | FPR test ↓ | FPR hard-neg ↓ | sufficiency ↑ | fact recall ↑ | wrong issue ↓ | halluc ↓ | p95 ms |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
LAT = json.load(open("results/latency_test.json")) if os.path.exists("results/latency_test.json") else {}
for v in order:
    t = S[v].get("test"); h = S[v].get("hardneg", {})
    if not t or not v[1:].isdigit(): continue
    lat = LAT.get(v, {}).get("p95")
    L.append(f"| {v}{' †' if int(v[1:]) >= 11 else ''} | {pct(t['protected'])} | {pct(t['leak_rate'])} | {pct(t['abuse_recall'])} | {pct(t['threat_recall'])} | {pct(t['threat_precision'])} | "
             f"{pct(t['false_positive_rate'])} | {pct(h.get('false_positive_rate'))} | {pct(t['sufficiency_toxic'])} | {pct(t['fact_recall_toxic'])} | "
             f"{pct(t.get('wrong_issue_rate'))} | {pct(t['halluc_rate'])} | {('%.0f' % lat) if lat is not None else '—'} |")
L += ["", "*protected* = abusive utterance detected **and** no abusive span reached the agent. *sufficiency* = every gold fact needed to resolve the complaint survives in what the agent receives (abusive utterances only). *FPR* = clean utterances that were rewritten. *halluc* = numbers in the agent's text that the caller never said. *wrong issue* = the agent's text asserts the wrong complaint type. *p95 ms* = clean per-utterance latency on Apple M5 (`eval/latency.py`), measured for the milestone versions. † = post-hoc: built after reading v10's test errors, so its test numbers are optimistic.", ""]
L += ["## LLM tier ablation (local Gemma 3 1B; Gemini needs an API key — see docs/DEPLOY.md)", "",
      "| config | split | protected | sufficiency | fact recall | halluc | LLM accepted / calls | reject reasons | LLM ms p50 |", "|---|---|---|---|---|---|---|---|---|"]
for v in ("v12", "llm_all", "llm_raw"):
    for sp in ("dev", "test"):
        t = S.get(v, {}).get(sp)
        if not t: continue
        llm = t.get("llm") or {}
        rej = ", ".join(f"{k[4:]} {n}" for k, n in llm.items() if k.startswith("rej_") and n)
        L.append(f"| {v} | {sp} | {pct(t['protected'])} | {pct(t['sufficiency_toxic'])} | {pct(t['fact_recall_toxic'])} | {pct(t['halluc_rate'])} | "
                 f"{llm.get('accepted', '—')} / {llm.get('calls', '—')} | {rej or ('unverified' if v == 'llm_raw' else '—')} | {('%.0f' % llm['ms_p50']) if llm.get('ms_p50') else '—'} |")
L.append("")
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
          "| STT | hop (s) | policy | RTF | leak@1s | leak@1.5s | leak@2s | leak@3s | leak@4s (accuracy ceiling) | false-muted s per clean utt @2s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in streams:
        s = json.load(open(p))["summary"]
        L.append(f"| {s['whisper']} | {s['chunk_s']} | {s['policy']} | {s['rtf_mean']:.2f} | "
                 f"{pct(s['word_leak@1.0'])} | {pct(s['word_leak@1.5'])} | {pct(s['word_leak@2.0'])} | {pct(s['word_leak@3.0'])} | {pct(s['word_leak@4.0'])} | {s['collateral_s_per_clean_utt@2.0']:.2f} |")
    L.append("\nRTF > 1 means the recogniser falls behind real time on this laptop CPU, so decisions arrive late and leak grows; the accuracy ceiling (leak@4s) separates recognition misses from lateness. The committed (LocalAgreement) row used an earlier, stricter mute-coverage criterion.")
open("results/RESULTS.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[:30]))
