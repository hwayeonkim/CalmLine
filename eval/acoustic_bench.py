"""CREMA-D: does cheap prosody detect hostile vocal arousal (anger) speaker-independently?
Also measures how much the look-ahead compressor reduces loudness of angry vs neutral speech."""
import io, json, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from calmline.audio.acoustic import features, LookaheadCompressor, frame_rms

CACHE = "data/external/cremad/features.json"


def load_rows():
    rows = []
    for split in ["validation", "test"]:
        t = pq.read_table(f"data/external/cremad/{split}.parquet").to_pylist()
        for r in t:
            name = os.path.basename(r["file"])[:-4]
            spk, sent, emo, inten = name.split("_")
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
            rows.append(dict(name=name, spk=spk, emo=emo, inten=inten, x=x.astype(np.float32), sr=sr))
    return rows


rows = load_rows()
if os.path.exists(CACHE):
    feats = json.load(open(CACHE))
else:
    t0 = time.time(); feats = {r["name"]: features(r["x"], r["sr"]) for r in rows}
    print("feature extraction ms/clip", round((time.time() - t0) * 1000 / len(rows), 1))
    json.dump(feats, open(CACHE, "w"))
spks = sorted({r["spk"] for r in rows})
rng = np.random.RandomState(0); rng.shuffle(spks)
test_spk = set(spks[: len(spks) // 3])
keys = list(next(iter(feats.values())).keys())
PROS = [k for k in keys if not k.startswith("mfcc")]


def matrix(rs, cols, spk_norm):
    X = np.array([[feats[r["name"]][c] for c in cols] for r in rs])
    if spk_norm:  # speaker-relative z-score: in a live call we compare to the caller's own first seconds
        by = {}
        for i, r in enumerate(rs): by.setdefault(r["spk"], []).append(i)
        for s, idx in by.items():
            mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-6
            X[idx] = (X[idx] - mu) / sd
    return X


tr = [r for r in rows if r["spk"] not in test_spk]; te = [r for r in rows if r["spk"] in test_spk]
ytr = np.array([r["emo"] == "ANG" for r in tr]); yte = np.array([r["emo"] == "ANG" for r in te])
res = {"n_train": len(tr), "n_test": len(te), "n_test_speakers": len(test_spk), "anger_share_test": float(yte.mean())}
for name, cols, norm in [("loudness_only", ["db_mean", "db_p95"], False), ("loudness_only+spknorm", ["db_mean", "db_p95"], True),
                         ("prosody", PROS, False), ("prosody+spknorm", PROS, True), ("prosody+mfcc+spknorm", keys, True)]:
    Xtr, Xte = matrix(tr, cols, norm), matrix(te, cols, norm)
    for mname, m in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                     ("gbm", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced"))]:
        if mname == "gbm" and name.startswith("loudness"): continue
        m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
        hi = np.array([r["inten"] in ("HI", "MD") for r in te])
        res[f"{name}/{mname}"] = dict(auc=roc_auc_score(yte, p), f1=f1_score(yte, p > 0.5), precision=precision_score(yte, p > 0.5), recall=recall_score(yte, p > 0.5),
                                      recall_anger_HI_MD=recall_score(yte[hi | ~yte], (p > 0.5)[hi | ~yte]))
        print(name, mname, {k: round(v, 3) for k, v in res[f"{name}/{mname}"].items()}, flush=True)
# compressor effect: loudness reduction on angry-HI vs neutral
comp = LookaheadCompressor()
red = {"ANG_HI": [], "NEU": []}
for r in rows:
    key = "ANG_HI" if (r["emo"] == "ANG" and r["inten"] == "HI") else ("NEU" if r["emo"] == "NEU" else None)
    if key is None: continue
    y = comp(r["x"])
    a, b = 20 * np.log10(frame_rms(r["x"]).max()), 20 * np.log10(frame_rms(y).max())
    red[key].append(a - b)
res["compressor_peak_reduction_db"] = {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in red.items()}
print(res["compressor_peak_reduction_db"])
os.makedirs("results/audio", exist_ok=True)
json.dump(res, open("results/audio/acoustic_cremad.json", "w"), indent=1, default=float)
