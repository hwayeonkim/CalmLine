"""Intent classifiers over the NYC 311 catalog."""
from __future__ import annotations
import re
from functools import lru_cache

import numpy as np

from .intents import INTENTS, UNKNOWN


class KeywordIntent:
    name = "keyword"

    def __init__(self):
        self.kw = {k: [re.compile(r"\b" + re.escape(w) + r"\b", re.I) for w in v["keywords"]] for k, v in INTENTS.items()}

    def __call__(self, text: str):
        best, bs = UNKNOWN, 0
        for k, pats in self.kw.items():
            s = sum(1 for p in pats if p.search(text))
            if s > bs:
                best, bs = k, s
        return best, float(bs)


@lru_cache(maxsize=2)
def _st(model_id):
    from sentence_transformers import SentenceTransformer
    import torch
    return SentenceTransformer(model_id, device="mps" if torch.backends.mps.is_available() else "cpu")


class EmbeddingIntent:
    """Nearest-prototype classifier (SetFit-style few-shot without fine-tuning).

    Prototypes = catalog examples + issue phrase. Score = max cosine similarity.
    Below `threshold` -> UNKNOWN (open-set rejection)."""

    def __init__(self, model_id="BAAI/bge-small-en-v1.5", threshold=0.6, extra_examples=None):
        self.model_id, self.threshold = model_id, threshold
        self.name = f"emb({model_id.split('/')[-1]})"
        self.labels, texts = [], []
        for k, v in INTENTS.items():
            for ex in v["examples"] + [v["issue"]] + (extra_examples or {}).get(k, []):
                self.labels.append(k)
                texts.append(ex)
        self.proto = self._enc(texts)

    def _enc(self, texts):
        return _st(self.model_id).encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)

    def scores(self, text: str):
        sims = self.proto @ self._enc([text])[0]
        best = {}
        for lab, s in zip(self.labels, sims):
            best[lab] = max(best.get(lab, -1), float(s))
        return best

    def __call__(self, text: str):
        sc = self.scores(text)
        lab = max(sc, key=sc.get)
        return (lab if sc[lab] >= self.threshold else UNKNOWN), sc[lab]
