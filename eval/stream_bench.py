"""Delay-buffer leak simulation (the 'broadcast delay' design).

For every utterance WAV we replay audio in real time to the streaming STT, run the abuse detector on
the transcript after every update, and record *when* each abusive word was flagged (wall-clock incl.
measured compute and queueing). With a playback delay D, the agent hears audio time t at wall time t+D,
so an abusive word starting at t leaks unless its mute decision exists by t+D.

policy 'committed' : act only on LocalAgreement-committed words (stable, slower)
policy 'tentative' : also act on the latest uncommitted hypothesis (faster, may over-mute)
Usage: python eval/stream_bench.py <split> <whisper_size> <chunk_s> [policy] [limit]"""
import json, os, sys, time
sys.path.insert(0, os.getcwd()); os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, soundfile as sf
from calmline.audio.stt import StreamingWhisper, WindowSpotter, offline_words, _norm
from calmline.toxicity import HybridDetector, ModelDetector, LexiconDetector
from calmline.textnorm import tokens

split, size, C = sys.argv[1], sys.argv[2], float(sys.argv[3])
policy = sys.argv[4] if len(sys.argv) > 4 else "committed"
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
DS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
EDGE = 0.35
PAD = float(os.getenv("MUTE_PAD", "0.25"))
gold = {json.loads(l)["id"]: json.loads(l) for l in open(f"data/gold/{split}.jsonl")}
det = HybridDetector(ModelDetector("unitary/toxic-bert"), LexiconDetector(extended="v4"), directed_gate=True)
det("warm up")
BIAS = os.getenv("BIAS", "1") == "1"
sw = WindowSpotter(size, float(os.getenv("WINDOW_S", "3.0")), bias_prompt=BIAS) if policy.startswith("window") else StreamingWhisper(size)
REF_CACHE = f"data/audio/{split}/ref_words.json"
ref_cache = json.load(open(REF_CACHE)) if os.path.exists(REF_CACHE) else {}


def ref_words(rid, path):
    if rid not in ref_cache:
        ref_cache[rid] = [(w.start, w.end, w.text) for w in offline_words(path, "small.en")]
    return ref_cache[rid]


def gold_toxic_idx(words, spans):
    toks = [(tokens(w[2]) or [""])[0] for w in words]
    idx = set()
    for sp in spans:
        st = tokens(sp)
        if not st: continue
        for i in range(len(toks)):
            if toks[i:i + len(st)] == st:
                idx.update(range(i, i + len(st)))
        if not any(toks[i:i + len(st)] == st for i in range(len(toks))):  # fuzzy: any single matching key token
            for i, t in enumerate(toks):
                if t in st and len(t) > 2 and t not in ("you", "your", "the", "are"):
                    idx.add(i)
    return idx


def flagged_words(words, extra=()):
    allw = list(words) + list(extra)
    text, offs = "", []
    for w in allw:
        if text: text += " "
        offs.append((len(text), len(text) + len(w.text))); text += w.text
    t0 = time.perf_counter(); r = det(text); dt = time.perf_counter() - t0
    out = set()
    for sp in r.spans:
        for i, (a, b) in enumerate(offs):
            if not (b <= sp.start or a >= sp.end): out.add(i)
    return [allw[i] for i in sorted(out)], dt, r


meta = json.load(open(f"data/audio/{split}/meta.json"))
records = []
for m in meta[:limit]:
    g = gold[m["id"]]
    x = sf.read(m["path"])[0].astype(np.float32); dur = len(x) / 16000
    ref = ref_words(m["id"], m["path"])
    gt = gold_toxic_idx(ref, g["toxic_spans"])
    pending = []
    sw.reset(); wall = 0.0; consumed = 0.0; decisions = []; computes = []
    while consumed < dur - 1e-6:
        wall = max(wall, min(dur, consumed + C))
        end = min(dur, wall)
        final = end >= dur - 1e-6
        t0 = time.perf_counter()
        got = sw.feed(x[int(consumed * 16000):int(end * 16000)], final=final)
        if policy.startswith("window"):
            fw, ddt, _ = flagged_words(got)
            if policy == "window_edge":
                # a flag on a word cut off by the window edge must be confirmed by the next window
                conf = []
                for w in fw:
                    if w.end < sw.window_end - EDGE or final or any(abs(p.start - w.start) < 0.3 and _norm(p.text) == _norm(w.text) for p in pending):
                        conf.append(w)
                pending = [w for w in fw if w not in conf]
                fw = conf
        else:
            extra = sw.prev if policy == "tentative" else []
            fw, ddt, _ = flagged_words(sw.committed, extra)
        c = time.perf_counter() - t0
        computes.append(c); wall += c; consumed = end
        for w in fw:
            decisions.append((wall, w.start - PAD, w.end + PAD))
    # final routing decision time (for calm text delivery)
    final_wall = wall
    hyp_text = sw.text() if hasattr(sw, "text") else ""
    rec = dict(id=m["id"], dur=dur, route=g["route"], n_toxic_words=len(gt), final_latency=final_wall - dur,
               rtf=sum(computes) / dur, hyp=hyp_text, ref=" ".join(w[2] for w in ref))
    for D in DS:
        leaked = 0
        for i in gt:
            s, e = ref[i][0], ref[i][1]
            ok = any(t <= s + D and (min(b, e) - max(a, s)) >= 0.5 * max(0.05, e - s) for t, a, b in decisions)
            leaked += (not ok)
        rec[f"leak_words@{D}"] = leaked
        # collateral: muted reference seconds on non-toxic words (decisions in time)
        coll = 0.0
        for j, w in enumerate(ref):
            if j in gt: continue
            if any(t <= w[0] + D and (min(b, w[1]) - max(a, w[0])) >= 0.5 * (w[1] - w[0]) for t, a, b in decisions): coll += w[1] - w[0]
        rec[f"collateral_s@{D}"] = coll
    records.append(rec)
    print(m["id"], g["route"], f"dur={dur:.1f} rtf={rec['rtf']:.2f} final_lat={rec['final_latency']:.2f}",
          "leak@2:", rec["leak_words@2.0"], "/", len(gt), flush=True)
json.dump(ref_cache, open(REF_CACHE, "w"))
tox = [r for r in records if r["n_toxic_words"] > 0]
summ = dict(split=split, whisper=size, chunk_s=C, policy=policy + ("+bias" if policy.startswith("window") and BIAS else ""), n=len(records), n_toxic_utts=len(tox),
            rtf_mean=float(np.mean([r["rtf"] for r in records])),
            final_latency_p50=float(np.percentile([r["final_latency"] for r in records], 50)),
            final_latency_p95=float(np.percentile([r["final_latency"] for r in records], 95)))
for D in DS:
    tw = sum(r["n_toxic_words"] for r in tox)
    summ[f"word_leak@{D}"] = sum(r[f"leak_words@{D}"] for r in tox) / max(1, tw)
    summ[f"utt_leak@{D}"] = float(np.mean([r[f"leak_words@{D}"] > 0 for r in tox])) if tox else 0
    clean = [r for r in records if r["route"] == "PASS"]
    summ[f"collateral_s_per_clean_utt@{D}"] = float(np.mean([r[f"collateral_s@{D}"] for r in clean])) if clean else 0
print(json.dumps(summ, indent=1))
os.makedirs("results/audio", exist_ok=True)
tag = f"{split}_{size}_c{C}_{policy}" + (f"_w{os.getenv('WINDOW_S', '3.0')}" + ("_bias" if BIAS else "") if policy.startswith("window") else "")
json.dump(dict(summary=summ, records=records), open(f"results/audio/stream_{tag}.json", "w"), indent=1)
