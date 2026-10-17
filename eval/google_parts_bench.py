"""Parts and finalization delay of the Google streaming path, sentence cut (pause only) vs. v18 clause cut.

Each demo sample call is fed to GoogleStreamingSTT in 300 ms chunks at real-time pace (as the caller page does).
Per part: delay = wall time the final arrived - wall time the part's last word ended.

--client google : the real Speech-to-Text API (needs Application Default Credentials)
--client proxy  : an offline stand-in that answers the same streaming protocol with faster-whisper (interims every
                  300 ms of audio, a final when the stream is half-closed). It exercises the cut logic end to end;
                  its delays are Whisper's, not Google's, so only compare the two modes with each other.
Usage: python eval/google_parts_bench.py [--client google|proxy] [--samples heat,threat] [--out results/x.json]"""
import argparse, glob, json, os, statistics as st, sys, threading, time
from datetime import timedelta
from types import SimpleNamespace as NS
sys.path.insert(0, os.getcwd()); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, soundfile as sf
from calmline.audio import google_stt as G

SR, CHUNK = 16000, 4800


class ProxyClient:
    """Answers streaming_recognize like speech_v1.SpeechClient, transcribing with faster-whisper."""
    _lock = threading.Lock()

    def __init__(self, interim="tiny.en", final="base.en"):
        from calmline.audio.stt import load_whisper
        self.mi, self.mf = load_whisper(interim), load_whisper(final)

    def _asr(self, m, x):
        with self._lock:
            segs, _ = m.transcribe(x, language="en", beam_size=1, word_timestamps=True, condition_on_previous_text=False)
            return [w for s in segs for w in (s.words or [])]

    def streaming_recognize(self, config, requests):
        # audio is drained by its own thread so a slow interim never holds back the half-close (as with Google,
        # interims describe the latest audio and are skipped, not queued, when the recognizer is behind)
        st_ = dict(buf=[], done=False)
        def drain():
            for req in requests:
                st_["buf"].append(np.frombuffer(req.audio_content, "<i2").astype(np.float32) / 32767)
            st_["done"] = True
        threading.Thread(target=drain, daemon=True).start()
        last = 0.0
        while not st_["done"]:
            buf = np.concatenate(st_["buf"]) if st_["buf"] else np.zeros(0, np.float32)
            if len(buf) / SR - last < 0.3:
                time.sleep(0.02); continue
            last = len(buf) / SR
            ws = self._asr(self.mi, buf)
            if ws:
                yield NS(results=[NS(is_final=False, stability=0.9, result_end_time=timedelta(seconds=last),
                                     alternatives=[NS(transcript="".join(w.word for w in ws).strip(), words=[])])])
        buf = np.concatenate(st_["buf"]) if st_["buf"] else np.zeros(0, np.float32)
        ws = self._asr(self.mf, buf) if len(buf) > 1600 else []
        if ws:
            yield NS(results=[NS(is_final=True, stability=1.0, result_end_time=timedelta(seconds=len(buf) / SR),
                                 alternatives=[NS(transcript="".join(w.word for w in ws).strip(),
                                                  words=[NS(word=w.word.strip(), start_time=timedelta(seconds=w.start),
                                                            end_time=timedelta(seconds=w.end)) for w in ws])])])


def run_call(path, clause, client):
    x, sr = sf.read(path, dtype="float32"); assert sr == SR
    g = G.GoogleStreamingSTT(clause=clause, client=client)
    parts, t_start = [], time.time()
    def collect():
        for e in g.poll():
            if e[0] == "final" and e[3]:
                _, t0, t1, text, words = e
                end = words[-1][1] if words else t1
                parts.append(dict(t0=round(t0, 2), t1=round(end, 2), text=text, n_words=len(text.split()),
                                  delay_s=round(time.time() - (t_start + end), 2)))
            elif e[0] == "error":
                raise RuntimeError(e[1])
    for k in range(0, len(x), CHUNK):
        due = t_start + (k + CHUNK) / SR
        time.sleep(max(0.0, due - time.time()))       # real-time arrival
        g.feed(x[k:k + CHUNK]); collect()
    g.close(wait=6.0); time.sleep(0.3); collect()
    return parts


def summary(parts):
    d = [p["delay_s"] for p in parts]
    return dict(parts=len(parts), words_per_part=round(st.mean(p["n_words"] for p in parts), 1) if parts else 0,
                median_delay_s=round(st.median(d), 2) if d else None, worst_delay_s=max(d) if d else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default="google", choices=["google", "proxy"])
    ap.add_argument("--samples", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    client = ProxyClient() if a.client == "proxy" else None
    if a.client == "google":
        ok, why = G.available()
        if not ok:
            sys.exit(f"no Google credentials: {why}")
    names = a.samples.split(",") if a.samples else [os.path.basename(p)[:-4] for p in sorted(glob.glob("app/static/samples/*.wav"))]
    res = {}
    for name in names:
        res[name] = {}
        for mode, clause in (("sentence", False), ("clause", True)):
            parts = run_call(f"app/static/samples/{name}.wav", clause, client)
            res[name][mode] = dict(summary(parts), detail=parts)
            s = res[name][mode]
            print(f"{name:9s} {mode:8s} parts={s['parts']:2d} words/part={s['words_per_part']:5.1f} "
                  f"median={s['median_delay_s']}s worst={s['worst_delay_s']}s", flush=True)
            for p in parts:
                print(f"    [{p['t0']:5.1f}-{p['t1']:5.1f}] +{p['delay_s']:.2f}s  {p['text']}", flush=True)
    tot = {}
    for mode in ("sentence", "clause"):
        allp = [p for n in res for p in res[n][mode]["detail"]]
        tot[mode] = dict(summary(allp), parts_per_call=round(len(allp) / len(res), 1))
        print(f"ALL {mode:8s} {tot[mode]}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(client=a.client, total=tot, calls=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
