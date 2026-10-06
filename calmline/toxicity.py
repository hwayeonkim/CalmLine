"""Abuse detectors.

LexiconDetector  — regex lexicon (v0 baseline, ~0 ms)
ModelDetector    — Jigsaw-trained transformer (Detoxify weights: unitary/toxic-bert or
                   unitary/unbiased-toxic-roberta), scored per clause for localisation
HybridDetector   — union of both; lexicon gives exact spans, model catches what the
                   lexicon misses (clause-level span)
"""
from __future__ import annotations
import re
import time
from functools import lru_cache

from . import lexicon
from .schema import Span, ToxicityResult

CLAUSE_RE = re.compile(r"[^.!?;]+[.!?;]*")


def clauses(text: str):
    """Split into sentence/comma clauses, returning (start, end, text)."""
    out = []
    for m in CLAUSE_RE.finditer(text):
        s0 = m.start()
        sub = m.group(0)
        pos = 0
        for part in re.split(r"(,\s+|\s+-\s+|\s+(?:but|and then|so)\s+)", sub):
            if not part:
                continue
            if re.fullmatch(r",\s+|\s+-\s+|\s+(?:but|and then|so)\s+", part):
                pos += len(part)
                continue
            st = s0 + pos
            en = st + len(part)
            if part.strip():
                lead = len(part) - len(part.lstrip())
                out.append((st + lead, en, part.strip()))
            pos += len(part)
    return out


class LexiconDetector:
    name = "lexicon"

    def __init__(self, include_mild=False):
        self.include_mild = include_mild

    def __call__(self, text: str) -> ToxicityResult:
        found = lexicon.find_spans(text, include_mild=self.include_mild)
        spans = [Span(s, e, cat, t) for s, e, cat, t in found if cat != "mild" or self.include_mild]
        threat = any(s.label in ("threat", "self_harm") for s in spans)
        toxic = any(s.label not in ("threat", "self_harm") for s in spans)
        return ToxicityResult(is_toxic=toxic or threat, is_threat=threat, score=1.0 if spans else 0.0,
                              threat_score=1.0 if threat else 0.0, spans=spans,
                              labels={s.label: 1.0 for s in spans})


@lru_cache(maxsize=4)
def _load(model_id: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).eval().to(device)
    return tok, mdl


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


class ModelDetector:
    """Sentence/clause-level Jigsaw classifier. Labels are mapped onto CalmLine categories."""

    def __init__(self, model_id="unitary/toxic-bert", tox_th=0.5, threat_th=0.5, device=None, per_clause=True):
        self.model_id, self.tox_th, self.threat_th = model_id, tox_th, threat_th
        self.device = device or _device()
        self.per_clause = per_clause
        self.name = model_id.split("/")[-1]

    def scores(self, texts):
        import torch
        tok, mdl = _load(self.model_id, self.device)
        with torch.no_grad():
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            probs = torch.sigmoid(mdl(**enc).logits).float().cpu().numpy()
        labels = [mdl.config.id2label[i] for i in range(probs.shape[1])]
        return [dict(zip(labels, p.tolist())) for p in probs]

    @staticmethod
    def tox_of(d):
        return d.get("toxic", d.get("toxicity", 0.0))

    def __call__(self, text: str) -> ToxicityResult:
        cl = clauses(text) if self.per_clause else [(0, len(text), text)]
        units = [text] + [c[2] for c in cl]
        sc = self.scores(units)
        whole, parts = sc[0], sc[1:]
        spans = []
        for (s, e, t), d in zip(cl, parts):
            if self.tox_of(d) >= self.tox_th or d.get("threat", 0) >= self.threat_th:
                lab = "threat" if d.get("threat", 0) >= self.threat_th else max(
                    (k for k in d if k in ("obscene", "insult", "identity_hate", "identity_attack", "sexual_explicit", "severe_toxic", "severe_toxicity")),
                    key=lambda k: d[k], default="insult")
                spans.append(Span(s, e, _MAP.get(lab, "insult"), t, self.tox_of(d)))
        tmax = max([self.tox_of(whole)] + [self.tox_of(d) for d in parts])
        thr = max([whole.get("threat", 0)] + [d.get("threat", 0) for d in parts])
        return ToxicityResult(is_toxic=tmax >= self.tox_th or thr >= self.threat_th, is_threat=thr >= self.threat_th,
                              score=tmax, threat_score=thr, spans=spans, labels=whole)


_MAP = {"obscene": "profanity", "insult": "insult", "identity_hate": "identity", "identity_attack": "identity",
        "sexual_explicit": "sexual", "severe_toxic": "insult", "severe_toxicity": "insult", "threat": "threat", "toxic": "insult",
        "toxicity": "insult"}


DIRECTED = re.compile(r"\b(?:you|your|you're|youre|ya|u|ur|yours|yourself|y'all|lady|sir|ma'am|miss|operator|agent|"
                      r"people like you|whoever|your (?:office|supervisor|boss|agency|department))\b", re.I)


class HybridDetector:
    """Union of lexicon and model.

    directed_gate (v3): a clause flagged *only* by the model must also be directed at a person
    (second-person/addressee reference) unless the model is very confident. Motivated by the
    abuse typology of Waseem et al. (2017): abuse aimed at the agent is 'directed'; figurative
    venting ("these rats are killing me") is not, and Jigsaw-trained models over-flag it."""

    def __init__(self, model: ModelDetector, lex: LexiconDetector | None = None, model_threat=True,
                 directed_gate=False, confident=0.9):
        self.model, self.lex = model, lex or LexiconDetector()
        self.model_threat, self.directed_gate, self.confident = model_threat, directed_gate, confident
        self.name = f"hybrid({model.name})"

    def __call__(self, text: str) -> ToxicityResult:
        a, b = self.lex(text), self.model(text)
        spans = list(a.spans)
        model_toxic = model_threat = False
        for sp in b.spans:  # add model clause spans only where lexicon found nothing inside that clause
            if any(x.start >= sp.start and x.end <= sp.end for x in a.spans):
                continue
            if self.directed_gate and sp.score < self.confident and not DIRECTED.search(sp.text):
                continue
            if sp.label == "threat" and not self.model_threat:
                sp.label = "insult"
            if sp.label == "threat":
                if self.directed_gate and not DIRECTED.search(sp.text) and not re.search(r"\b(?:i'?ll|i will|i'?m (?:gonna|going to)|gonna)\b", sp.text, re.I):
                    sp.label = "insult"
                else:
                    model_threat = True
            model_toxic = True
            spans.append(sp)
        if not self.directed_gate:
            model_toxic, model_threat = b.is_toxic, b.is_threat
        threat = a.is_threat or (self.model_threat and model_threat)
        return ToxicityResult(is_toxic=a.is_toxic or model_toxic or threat, is_threat=threat,
                              score=max(a.score, b.score), threat_score=max(a.threat_score, b.threat_score),
                              spans=sorted(spans, key=lambda s: s.start), labels=b.labels)
