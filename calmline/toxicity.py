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


# v11 (post-hoc): 311 *topic* vocabulary that Jigsaw-trained models read as insults ("coming out of the garbage" -> 0.99).
# Replaced by neutral words before model scoring only; insults that use these words ("you're garbage") are caught by
# the lexicon. Same idea as identity-term masking for unintended bias (Dixon et al., 2018).
TOPIC = re.compile(r"\b(?:garbage|trash|rubbish|litter|sewage|sewer|filth|filthy|disgusting|dump(?:ed|ing)?|rats?|mice|roach(?:es)?|cockroach(?:es)?|"
                   r"rodents?|pests?|vermin|mold|mould|feces|poop|urine|piss(?:ing)?)\b", re.I)


def neutralize(text: str) -> str:
    return TOPIC.sub("things", text)


class LexiconDetector:
    name = "lexicon"

    def __init__(self, include_mild=False, extended=True):
        self.include_mild, self.extended = include_mild, extended

    def __call__(self, text: str) -> ToxicityResult:
        found = lexicon.find_spans(text, include_mild=self.include_mild, extended=self.extended)
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

    def __init__(self, model_id="unitary/toxic-bert", tox_th=0.5, threat_th=0.5, device=None, per_clause=True, neutralize_topics=False):
        self.model_id, self.tox_th, self.threat_th = model_id, tox_th, threat_th
        self.neutralize_topics = neutralize_topics
        self.device = device or _device()
        self.per_clause = per_clause
        self.name = model_id.split("/")[-1]

    _memo: dict = {}

    def scores(self, texts):
        import os
        if os.getenv("CALMLINE_NO_MEMO") == "1":
            return self._scores(texts)
        key = (self.model_id, tuple(texts))
        if key in ModelDetector._memo:
            return ModelDetector._memo[key]
        res = self._scores(texts)
        if len(ModelDetector._memo) < 200000:
            ModelDetector._memo[key] = res
        return res

    def _scores(self, texts):
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
        if self.neutralize_topics:
            units = [neutralize(u) for u in units]
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


class SafetyNetDetector:
    """Wraps a detector and adds a zero-shot NLI check for threats / self-harm only (v6).

    Threat recall must be ~100%, but lexicons and Jigsaw models miss novel phrasings
    ("I'll put a bullet in someone"). NLI runs only on clauses not already flagged as threats."""

    def __init__(self, base, nli=None, th=0.9):
        self.base = base
        self.nli = nli or NLIDetector(cats=["threat", "self_harm"], th={"threat": th, "self_harm": th})
        self.name = f"{base.name}+nli"

    def __call__(self, text):
        r = self.base(text)
        if r.is_threat:
            return r
        n = self.nli(text)
        if n.is_threat:
            r.spans = sorted(r.spans + [s for s in n.spans if s.label in ("threat", "self_harm")], key=lambda s: s.start)
            r.is_threat = r.is_toxic = True
            r.threat_score = max(r.threat_score, n.threat_score)
        return r


# v8: model-only flags in these contexts are venting / concern, not abuse (hyperbole: Troiano et al., 2018)
FIGURATIVE = re.compile(r"\b(?:kill(?:ing|s)?|murder(?:ing)?|death|dying|die|crazy|insane|nightmare)\b(?:\W+\w+){0,3}?\W+(?:me|us|myself|ourselves)\b|"
                        r"\bto death\b|\b(?:i'?m|we'?re|i am|we are) dying\b|\b(?:wake|raise) the dead\b|\bdeath ?trap\b|\bsmells? like death\b", re.I)
CONCERN = re.compile(r"\b(?:worried|afraid|scared|terrified|concerned|fear(?:ful)?|frightened|before (?:someone|somebody|anyone)|"
                     r"could (?:get|be)|might (?:get|be)|going to get (?:hurt|killed)|someone (?:could|will|is going to) (?:die|get))\b", re.I)
FIRST_PERSON_INTENT = re.compile(r"\b(?:i'?ll|i will|i'?m (?:gonna|going to)|im gonna|gonna|i swear)\b", re.I)


class HybridDetector:
    """Union of lexicon and model.

    directed_gate (v3): a clause flagged *only* by the model must also be directed at a person
    (second-person/addressee reference) unless the model is very confident. Motivated by the
    abuse typology of Waseem et al. (2017): abuse aimed at the agent is 'directed'; figurative
    venting ("these rats are killing me") is not, and Jigsaw-trained models over-flag it."""

    def __init__(self, model: ModelDetector, lex: LexiconDetector | None = None, model_threat=True,
                 directed_gate=False, confident=0.9, threat_bypass=False, context_filter=False):
        self.model, self.lex = model, lex or LexiconDetector()
        self.model_threat, self.directed_gate, self.confident = model_threat, directed_gate, confident
        self.threat_bypass = threat_bypass   # v7: threats escalate even when aimed at a third party
        self.context_filter = context_filter # v8: drop model-only flags in figurative / concern contexts
        self.name = f"hybrid({model.name})"

    def __call__(self, text: str) -> ToxicityResult:
        a, b = self.lex(text), self.model(text)
        spans = list(a.spans)
        model_toxic = model_threat = False
        for sp in b.spans:  # add model clause spans only where lexicon found nothing inside that clause
            if any(x.start >= sp.start and x.end <= sp.end for x in a.spans):
                continue
            if self.directed_gate and sp.score < self.confident and not DIRECTED.search(sp.text) \
                    and not (self.threat_bypass and sp.label == "threat"):
                continue
            if self.context_filter and not FIRST_PERSON_INTENT.search(sp.text) and \
                    (FIGURATIVE.search(sp.text) or (CONCERN.search(sp.text) and not DIRECTED.search(sp.text))):
                continue
            if sp.label == "threat" and not self.model_threat:
                sp.label = "insult"
            if sp.label == "threat":
                if self.directed_gate and not self.threat_bypass and not DIRECTED.search(sp.text) and not re.search(r"\b(?:i'?ll|i will|i'?m (?:gonna|going to)|gonna)\b", sp.text, re.I):
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


HYPOTHESES = {
    "threat": "The speaker threatens to physically harm someone.",
    "self_harm": "The speaker says they want to hurt or kill themselves.",
    "sexual": "The speaker makes a sexual or flirtatious comment to the listener.",
    "insult": "The speaker insults or demeans the listener.",
    "identity": "The speaker attacks the listener's ethnicity, nationality, accent, or language.",
}


@lru_cache(maxsize=2)
def _load_nli(model_id, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).eval().to(device)
    return tok, mdl


class NLIDetector:
    """Zero-shot NLI abuse detector (Yin et al., 2019 'entailment as zero-shot classification').

    Each clause is the premise; each CalmLine category is a natural-language hypothesis.
    No abuse-specific training data -> does not depend on any lexicon, so it is used as a
    semantic safety net for the categories lexicons miss (implicit threats, harassment)."""

    def __init__(self, model_id="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", th=None, cats=None, device=None):
        self.model_id = model_id
        self.th = {"threat": 0.9, "self_harm": 0.9, "sexual": 0.9, "insult": 0.95, "identity": 0.9}
        self.th.update(th or {})
        self.cats = cats or list(HYPOTHESES)
        self.device = device or _device()
        self.name = "nli(" + model_id.split("/")[-1] + ")"

    def clause_scores(self, units):
        import torch
        tok, mdl = _load_nli(self.model_id, self.device)
        pairs = [(u, HYPOTHESES[c]) for u in units for c in self.cats]
        with torch.no_grad():
            enc = tok([p for p, _ in pairs], [h for _, h in pairs], return_tensors="pt", padding=True, truncation=True, max_length=160).to(self.device)
            logits = mdl(**enc).logits.float()
        lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
        ent = lab.get("entailment", 0)
        probs = torch.softmax(logits, -1)[:, ent].cpu().numpy() if logits.shape[1] > 1 else torch.sigmoid(logits[:, 0]).cpu().numpy()
        out = []
        for i in range(len(units)):
            out.append({c: float(probs[i * len(self.cats) + j]) for j, c in enumerate(self.cats)})
        return out

    def __call__(self, text: str) -> ToxicityResult:
        cl = clauses(text) or [(0, len(text), text)]
        sc = self.clause_scores([c[2] for c in cl])
        spans = []
        for (s, e, t), d in zip(cl, sc):
            hits = [c for c in self.cats if d[c] >= self.th[c]]
            if hits:
                lab = "threat" if "threat" in hits else ("self_harm" if "self_harm" in hits else max(hits, key=lambda c: d[c]))
                spans.append(Span(s, e, lab, t, d[lab]))
        thr = max([max(d.get("threat", 0), d.get("self_harm", 0)) for d in sc] + [0])
        tox = max([max(d[c] for c in self.cats) for d in sc] + [0])
        is_thr = any(sp.label in ("threat", "self_harm") for sp in spans)
        return ToxicityResult(is_toxic=bool(spans), is_threat=is_thr, score=tox, threat_score=thr, spans=spans)
