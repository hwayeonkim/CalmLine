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
    issue = INTENTS[intent]["issue"] if intent in INTENTS else "a problem"
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
    s = [f"Hello, I'm calling to report {issue}" + (f" at {_join(loc)}" if loc else "") + "."]
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


class HybridRenderer:
    """Template header for tone + verbatim detail clauses so facts survive (extractive detox)."""
    name = "hybrid"

    def __init__(self, min_new_tokens=2):
        self.min_new = min_new_tokens

    def __call__(self, text, intent, slots, tox):
        head = compose(intent, slots)
        have = set(content_tokens(head))
        details = []
        for s, e, c in clauses(text):
            if any(not (sp.end <= s or sp.start >= e) for sp in tox.spans):
                # clause contains abuse: keep only if, after masking, real content remains
                inner = [sp for sp in tox.spans if not (sp.end <= s or sp.start >= e)]
                if any(sp.label == "threat" for sp in inner):
                    continue
                c = mask(text[s:e], [type(sp)(sp.start - s, sp.end - s, sp.label, sp.text) for sp in inner])
            c = c.strip(" ,.!?")
            if not c or _RHETORICAL.match(c) or c.endswith("?") and len(content_tokens(c)) < 6:
                continue
            new = [w for w in content_tokens(c) if w not in have]
            if len(new) >= self.min_new:
                details.append(c[0].upper() + c[1:])
                have |= set(new)
        if details:
            head += " Details: " + ". ".join(details) + "."
        return head
