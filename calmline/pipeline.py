"""CalmLine text pipeline: route -> (intent, slots) -> 3-tier render ladder."""
from __future__ import annotations
import time

from .intents import UNKNOWN
from .schema import Decision
from . import slots as slotmod

ALERT = "[SAFETY ALERT: the caller made a threatening or self-harm statement. A supervisor has been notified.] "


class Pipeline:
    def __init__(self, detector, intent_clf, renderer, fallback=None, llm=None, name="pipeline", intent_min=0.0,
                 mask_for_intent=True, unknown_mode="mask"):
        self.detector, self.intent_clf, self.renderer = detector, intent_clf, renderer
        self.fallback = fallback          # renderer used when intent is UNKNOWN (tier 3 mask by default)
        self.llm = llm                    # optional tier-2 rewriter
        self.name, self.intent_min, self.mask_for_intent = name, intent_min, mask_for_intent
        self.unknown_mode = unknown_mode  # "mask": tier-3 masked transcript; "generic": template with generic header

    def __call__(self, text: str) -> Decision:
        t = {}
        t0 = time.perf_counter()
        tox = self.detector(text)
        t["detect"] = (time.perf_counter() - t0) * 1000
        route = "ESCALATE" if tox.is_threat else ("CALM" if tox.is_toxic else "PASS")
        d = Decision(route=route, delivered=text, abuse_types=sorted({s.label for s in tox.spans}), anger=tox.score)
        if route == "PASS":
            d.timings_ms = t
            return d
        t1 = time.perf_counter()
        from .render import mask
        clean = mask(text, tox.spans) if self.mask_for_intent else text
        intent, isc = self.intent_clf(clean or text)
        t["intent"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        sl = slotmod.extract(text)
        t["slots"] = (time.perf_counter() - t2) * 1000
        d.intent, d.intent_score, d.slots = intent, isc, sl
        t3 = time.perf_counter()
        if intent != UNKNOWN and isc >= self.intent_min:
            d.tier, out = 1, self.renderer(text, intent, sl, tox)
        elif self.unknown_mode == "generic" and self.llm is None:
            d.tier, out = 3, self.renderer(text, UNKNOWN, sl, tox)
        elif self.llm is not None:
            d.tier, out = 2, self.llm(text, intent, sl, tox)
        else:
            from .render import MaskRenderer
            d.tier, out = 3, (self.fallback or MaskRenderer())(text, intent, sl, tox)
        t["render"] = (time.perf_counter() - t3) * 1000
        d.delivered = (ALERT if route == "ESCALATE" else "") + out
        d.timings_ms = t
        return d
