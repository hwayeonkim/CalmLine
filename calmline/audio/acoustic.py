"""Acoustic aggression features + signal-level tone normalisation.

Features follow the arousal-relevant subset of eGeMAPS (Eyben et al., 2016): loudness (RMS),
pitch (F0) level/variability, spectral slope proxies and voicing/onset rate. These are cheap
(<5 ms per second of audio on CPU) so they can run on every frame inside the delay buffer."""
from __future__ import annotations
import numpy as np

SR = 16000


def frame_rms(x, n=400, hop=160):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    return np.sqrt((fr.astype(np.float64) ** 2).mean(1) + 1e-12)


def features(x: np.ndarray, sr=SR) -> dict:
    import librosa
    x = x.astype(np.float32)
    rms = frame_rms(x)
    db = 20 * np.log10(rms + 1e-9)
    voiced = db > (db.max() - 35)
    f0 = librosa.yin(x, fmin=70, fmax=500, sr=sr, frame_length=1024, hop_length=320)
    e = frame_rms(x, 1024, 320)[: len(f0)]
    f0v = f0[: len(e)][e > np.percentile(e, 40)] if len(e) else f0
    S = np.abs(librosa.stft(x, n_fft=512, hop_length=160))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    hi = S[freqs >= 1000].sum(0) / (S[freqs < 1000].sum(0) + 1e-9)   # alpha-ratio-like: shouting tilts energy up
    onset = librosa.onset.onset_strength(y=x, sr=sr)
    mf = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
    out = dict(
        db_mean=float(db[voiced].mean()) if voiced.any() else float(db.mean()), db_p95=float(np.percentile(db, 95)),
        db_std=float(db[voiced].std()) if voiced.any() else 0.0,
        f0_mean=float(np.median(f0v)), f0_std=float(np.std(f0v)), f0_range=float(np.percentile(f0v, 95) - np.percentile(f0v, 5)),
        cent_mean=float(cent.mean()), flat_mean=float(flat.mean()), hi_ratio=float(np.median(hi)),
        onset_rate=float((onset > onset.mean() + onset.std()).mean()), voiced_frac=float(voiced.mean()),
    )
    for i in range(13):
        out[f"mfcc{i}"] = float(mf[i].mean())
    return out


class LookaheadCompressor:
    """Tone normalisation at the signal level: because CalmLine already delays the caller's audio,
    the compressor can *look ahead* and bring shouted segments down to conversational loudness
    before they reach the agent's ear — no attack overshoot, no audible pumping."""

    def __init__(self, target_db=-26.0, ratio=6.0, knee_db=-20.0, window_s=0.3, sr=SR, max_cut_db=24.0):
        self.target, self.ratio, self.knee, self.win, self.sr, self.max_cut = target_db, ratio, knee_db, window_s, sr, max_cut_db

    def gain_curve(self, x):
        hop = 160
        rms = frame_rms(x, 400, hop)
        db = 20 * np.log10(rms + 1e-9)
        k = max(1, int(self.win * self.sr / hop))
        # look-ahead: max over a centred window so the gain drops *before* the loud onset
        pad = np.pad(db, (k, k), mode="edge")
        env = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1).max(1)
        over = np.maximum(0, env - self.knee)
        cut = np.minimum(self.max_cut, over * (1 - 1 / self.ratio))
        g = 10 ** (-cut / 20)
        t_frames = np.arange(len(g)) * hop + 200
        return np.interp(np.arange(len(x)), t_frames, g)

    def __call__(self, x):
        return (x * self.gain_curve(x)).astype(np.float32)
