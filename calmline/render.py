"""Renderers: turn an abusive utterance into what the agent receives.

TemplateRenderer   — tier 1: pre-written calm phrase per intent + verbatim slots (no generation).
MaskRenderer       — tier 3: original transcript with abusive spans removed.
HybridRenderer     — template header + extractive 'details' (clauses with no abuse, not already covered).
"""
from __future__ import annotations
import re

from .intents import INTENTS, UNKNOWN
from .textnorm import content_tokens
from .toxicity import clauses


def _join(xs):
    xs = [x for x in xs if x]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _strip_lead(s: str):
    s = s.strip(" ,.")
    s = re.sub(r"^(?:at|on|in|near|by|of|the corner of|corner of)\s+", "", s, flags=re.I)
    return s


_REQ_LEAD = re.compile(r"^(?:i want|i need|i'd like|can you|could you|please|can someone|can somebody|need someone to|somebody needs to|someone needs to)\s+", re.I)


def compose(intent: str, slots: dict) -> str:
    issue = INTENTS[intent]["issue"] if intent in INTENTS else "an issue that may not fit a standard category"
    loc = []
    if slots.get("apartment"):
        loc.append("apartment " + re.sub(r"^(?:apartment|apt\.?|unit|in|from|at|,)\s*", "", slots["apartment"][0], flags=re.I).strip())
    if slots.get("floor"):
        loc.append("on the " + slots["floor"][0])
    for k in ("address", "cross_street", "street"):
        for v in slots.get(k, [])[:2]:
            loc.append(_strip_lead(v))
    if slots.get("borough"):
        loc.append(slots["borough"][0].title())
    verb = "report" if intent in INTENTS else "ask about"
    if intent == "__NEUTRAL__":
        s = ["Hello, I'm calling about a problem" + (f" at {_join(loc)}" if loc else "") + "."]
    else:
        s = [f"Hello, I'm calling to {verb} {issue}" + (f" at {_join(loc)}" if loc else "") + "."]
    veh = slots.get("vehicle", [])[-1:] + [("plate " + p if not p.lower().startswith(("plate", "license")) else p) for p in slots.get("plate", [])]
    if veh:
        s.append(f"The vehicle involved: {_join(veh)}.")
    when = slots.get("time", []) + slots.get("duration", [])
    if when:
        s.append(f"Timing: {_join(when)}.")
    if slots.get("vulnerable"):
        s.append(f"Please note: {_join(slots['vulnerable'])}.")
    if slots.get("prior_contact"):
        s.append(f"I have contacted you before ({_join(slots['prior_contact'])}).")
    if slots.get("reference"):
        s.append(f"My reference is {_join(slots['reference'])}.")
    if slots.get("request"):
        reqs = [_REQ_LEAD.sub("", r).strip() for r in slots["request"]]
        s.append(f"I'm asking you to please {_join(reqs)}.")
    return " ".join(s)


class TemplateRenderer:
    name = "template"

    def __call__(self, text, intent, slots, tox):
        return compose(intent, slots)


def mask(text: str, spans, repl="") -> str:
    out, last = [], 0
    for sp in sorted(spans, key=lambda s: s.start):
        if sp.start < last:
            last = max(last, sp.end)
            continue
        out.append(text[last:sp.start])
        out.append(repl)
        last = sp.end
    out.append(text[last:])
    s = "".join(out)
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    s = re.sub(r"([,.!?])\1+", r"\1", s)
    s = re.sub(r"^[\s,.!?]+", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


class MaskRenderer:
    name = "mask"

    def __call__(self, text, intent, slots, tox):
        return mask(text, tox.spans, "")


_RHETORICAL = re.compile(r"^(?:are you|do you|did you|is it|what kind|who the|what the|how many times|can you understand|"
                         r"you hear me|nobody (?:gives|ever)|this is (?:bullshit|ridiculous|insane)|oh|god|jesus|listen|look|"
                         r"i swear|whatever|fix it|do something|get (?:off|your)|write (?:this|it) down|anyway|haha|mmm|"
                         r"okay so|ok so|ok listen|hey)\b", re.I)


def _slot_spans(text):
    from .slots import spans
    return [(a, b, n) for a, b, n in spans(text) if n not in ("time", "duration", "request", "prior_contact")]


class HybridRenderer:
    """Template header for tone + verbatim detail clauses so facts survive (extractive detox).

    verify (v3): every candidate detail clause is re-scored by the detector after masking and
    dropped if it is still abusive (render-then-verify). For threats, clauses that contained any
    abuse are dropped entirely rather than masked."""
    name = "hybrid"

    def __init__(self, min_new_tokens=2, verify=None, drop_residue=False, escalate_no_details=False):
        self.escalate_no_details = escalate_no_details  # v11: a threat call gets header + slots only
        self.min_new = min_new_tokens
        self.verify = verify
        self.drop_residue = drop_residue  # v4: drop abusive clauses that carry no slot/fact content

    def __call__(self, text, intent, slots, tox):
        head = compose(intent, slots)
        if self.escalate_no_details and tox.is_threat:
            return head
        have = set(content_tokens(head)) if intent != "__NEUTRAL__" else set(content_tokens(head)) - {"problem"}
        if intent == "__NEUTRAL__":
            have = set()   # neutral header: keep every informative clause, even if slots repeat it
        details = []
        for s, e, c in clauses(text):
            if any(not (sp.end <= s or sp.start >= e) for sp in tox.spans):
                # clause contains abuse: keep only if, after masking, real content remains
                inner = [sp for sp in tox.spans if not (sp.end <= s or sp.start >= e)]
                if any(sp.label in ("threat", "self_harm") for sp in inner) or (tox.is_threat and self.verify is not None):
                    continue
                c = mask(text[s:e], [type(sp)(sp.start - s, sp.end - s, sp.label, sp.text) for sp in inner])
                if self.drop_residue:
                    has_slot = any(ss >= s and se <= e for ss, se, _ in _slot_spans(text))
                    cover = sum(min(sp.end, e) - max(sp.start, s) for sp in inner) / max(1, e - s)
                    if not has_slot and (cover >= 0.4 or len(content_tokens(c)) < 5):
                        continue
            c = c.strip(" ,.!?")
            if not c or _RHETORICAL.match(c) or c.endswith("?") and len(content_tokens(c)) < 6:
                continue
            new = [w for w in content_tokens(c) if w not in have]
            if len(new) >= self.min_new and self.verify is not None:
                v = self.verify(c)
                if v.is_toxic or v.is_threat:
                    continue
            if len(new) >= self.min_new:
                details.append(c[0].upper() + c[1:])
                have |= set(new)
        if details:
            head += " Details: " + ". ".join(details) + "."
        return head


class ConfidenceGatedRenderer(HybridRenderer):
    """v9: only *assert* the issue type in the agent's text when the intent classifier is confident.

    A wrong template header is a silent hallucination ("noise complaint" for a sewer backup). Below
    `assert_th` the header is neutral and the caller's own (cleaned) words carry the issue; the intent is
    still shown to the agent as a separate, clearly-labelled suggestion chip."""
    name = "gated"

    def __init__(self, assert_th=0.75, **kw):
        super().__init__(**kw)
        self.assert_th = assert_th
        self.last_score = 1.0

    def __call__(self, text, intent, slots, tox, score=None):
        sc = self.last_score if score is None else score
        if sc < self.assert_th:
            intent = "__NEUTRAL__"
        return super().__call__(text, intent, slots, tox)
