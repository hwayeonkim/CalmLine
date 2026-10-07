"""Confidence-gated offline relay (v20): handle a part without an LLM when the abuse is fully explained by the
lexicon and removing it leaves a clean, undirected sentence. Otherwise defer to Gemini (LLM cascade, cf. FrugalGPT)."""
from __future__ import annotations
import re

from .render import mask
from .textnorm import content_tokens
from .toxicity import clauses, DIRECTED

INSULT_NOUNS = re.compile(r"^(?:asshole|assholes|idiot|idiots|moron|morons|bitch|bastard|bastards|jerk|jerks|dickhead|dumbass|dumbasses|"
                          r"motherfuckers?|clowns?|losers?|scumbags?|jackass|dipshit|imbeciles?|twits?)$", re.I)
_SARC = re.compile(r"\b(?:great job|good job|nice work|well done|thanks for nothing|as usual|as always|must be nice|bless|genius|"
                   r"no offense|with a brain|wow|obviously|clearly)\b", re.I)


def _replace_insult_nouns(text, spans):
    """'some asshole parked' -> 'someone parked', 'this idiot' -> 'this person'."""
    keep, out = [], []
    for sp in spans:
        word = text[sp.start:sp.end].strip()
        pre = text[max(0, sp.start - 6):sp.start].lower()
        if INSULT_NOUNS.match(word) and re.search(r"\b(?:some|a|an|this|that|the)\s+$", pre):
            out.append((sp.start, sp.end, "one" if pre.rstrip().endswith("some") else "person"))
        else:
            keep.append(sp)
    for s, e, rep in sorted(out, reverse=True):
        if rep == "one":
            j = text.lower().rfind("some", 0, s)
            text = text[:j] + ("Someone" if text[j:j + 1].isupper() else "someone") + text[e:]
            shift = (e - s) + (s - j) - len("someone")
            keep = [type(k)(k.start - shift if k.start > s else k.start, k.end - shift if k.start > s else k.end, k.label, k.text, k.score) for k in keep]
        else:
            text = text[:s] + rep + text[e:]
            shift = (e - s) - len(rep)
            keep = [type(k)(k.start - shift if k.start > s else k.start, k.end - shift if k.start > s else k.end, k.label, k.text, k.score) for k in keep]
    return text, keep


# Profanity used as an intensifier: deleting the word leaves a fluent, complete sentence.
INTENSIFIER = re.compile(r"^(?:fucking|fuckin|freaking|frigging|effing|goddamn(?:ed)?|goddam|god damn|damn(?:ed)?|bloody|shitty|"
                         r"damn fucking|bloody fucking|motherfucking|fucking hell|goddammit)$", re.I)


def with_mild(text, tox):
    from .lexicon import COMPILED
    from .schema import Span
    spans = list(tox.spans)
    for p in COMPILED["mild"]:
        for m in p.finditer(text):
            if not any(s.start <= m.start() and s.end >= m.end() for s in spans):
                spans.append(Span(m.start(), m.end(), "mild", m.group(0)))
    return sorted(spans, key=lambda s: s.start)


_FILLER = set("""you your you're youre people something or are is was what kind of this that so me i just even it a an the and to do does did
be been being going gonna get got man lady guys all every single too hard for with like really now here there yeah ok okay listen look hey
oh god jesus well right sure dude bro sir maam ma'am miss mister buddy pal again can could would will shall must how why who
whatever seriously honestly literally always never ever""".split())


def offline_withhold(text, tox):
    """Withhold immediately only when, after removing every abusive span, nothing informative is left: no slot,
    no complaint keyword, fewer than two meaningful words. Anything else goes to Gemini (never drop facts)."""
    from .slots import extract
    from .intent_clf import KeywordIntent
    from .intents import UNKNOWN
    if tox.is_threat or not tox.spans or any(sp.score < 1.0 for sp in tox.spans):
        return False
    rest = mask(text, with_mild(text, tox))
    if any(extract(rest).values()):
        return False
    if KeywordIntent()(rest)[0] != UNKNOWN:
        return False
    meaningful = [w for w in content_tokens(rest) if w not in _FILLER]
    return len(meaningful) < 2


def offline_calm(text, tox, detector):
    """Returns (message, confident). Only intensifier profanity is handled offline; anything else goes to Gemini."""
    if tox.is_threat:
        return None, False
    if offline_withhold(text, tox):
        return "", True
    spans_all = with_mild(text, tox)

    def third_party_insult(sp):
        word = text[sp.start:sp.end].strip()
        pre = text[max(0, sp.start - 6):sp.start].lower()
        return bool(INSULT_NOUNS.match(word) and re.search(r"\b(?:some|this|that|the)\s+$", pre)
                    and not re.search(r"\byou\b", text[max(0, sp.start - 12):sp.end + 6], re.I))

    if not tox.spans or not all(INTENSIFIER.match(text[sp.start:sp.end].strip()) or sp.label == "mild" or third_party_insult(sp)
                                for sp in spans_all):
        return None, False
    tox = type(tox)(tox.is_toxic, tox.is_threat, tox.score, tox.threat_score, spans_all, tox.labels)
    if any(sp.score < 1.0 for sp in tox.spans):          # a model-only flag: not explained by the lexicon -> ask Gemini
        return None, False
    t2, spans = _replace_insult_nouns(text, tox.spans)
    kept = []
    for s, e, c in clauses(t2):
        inner = [sp for sp in spans if not (sp.end <= s or sp.start >= e)]
        if inner:
            rest = mask(t2[s:e], [type(sp)(max(0, sp.start - s), min(e, sp.end) - s, sp.label, sp.text, sp.score) for sp in inner])
            c = rest                                       # intensifiers only: delete the word, never the clause
        kept.append(c.strip(" ,"))
    msg = " ".join(k[0].upper() + k[1:] if k else k for k in kept if k).strip()
    msg = re.sub(r"\s+([,.!?])", r"\1", msg)
    msg = re.sub(r"\s{2,}", " ", msg)
    if not msg:
        return None, False
    if not re.search(r"[.!?]$", msg):
        msg += "."
    if DIRECTED.search(msg) or _SARC.search(text):           # still talking to / about the agent -> Gemini decides
        return None, False
    if msg and detector(msg).is_toxic:
        return None, False
    return msg, True
