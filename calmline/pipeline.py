"""CalmLine text pipeline: route -> (intent, slots) -> 3-tier render ladder."""
from __future__ import annotations
import time

from .intents import UNKNOWN
from .schema import Decision
from . import slots as slotmod

ALERT = "[SAFETY ALERT: the caller made a threatening or self-harm statement. A supervisor has been notified.] "


class Pipeline:
    def __init__(self, detector, intent_clf, renderer, fallback=None, llm=None, name="pipeline", intent_min=0.0,
                 mask_for_intent=True, unknown_mode="mask", mask_mild=False, mask_slots=False, understanding=None,
                 understand_all=False, checker=None, gemini_detects=False):
        self.detector, self.intent_clf, self.renderer = detector, intent_clf, renderer
        self.fallback = fallback          # renderer used when intent is UNKNOWN (tier 3 mask by default)
        self.llm = llm                    # optional tier-2 rewriter
        self.name, self.intent_min, self.mask_for_intent = name, intent_min, mask_for_intent
        self.unknown_mode = unknown_mode
        self.mask_mild = mask_mild
        self.mask_slots = mask_slots
        self.understanding, self.understand_all, self.checker = understanding, understand_all, checker
        self.gemini_detects = gemini_detects
        self.context = ""        # v4: once a call is being rewritten, also strip mild profanity  # "mask": tier-3 masked transcript; "generic": template with generic header

    def __call__(self, text: str) -> Decision:
        t = {}
        t0 = time.perf_counter()
        tox = self.detector(text)
        t["detect"] = (time.perf_counter() - t0) * 1000
        route = "ESCALATE" if tox.is_threat else ("CALM" if tox.is_toxic else "PASS")
        d = Decision(route=route, delivered=text, abuse_types=sorted({s.label for s in tox.spans}), anger=tox.score)
        d.debug["spans"] = [(s.start, s.end, s.label) for s in tox.spans]
        if route == "PASS" and not self.understand_all:
            d.timings_ms = t
            return d
        if self.understanding is not None:
            return self._gemini_path(text, tox, d, t)
        if self.mask_mild:
            from .lexicon import COMPILED
            from .schema import Span
            for p in COMPILED["mild"]:
                for m in p.finditer(text):
                    if not any(sp.start <= m.start() and sp.end >= m.end() for sp in tox.spans):
                        tox.spans.append(Span(m.start(), m.end(), "mild", m.group(0)))
        t1 = time.perf_counter()
        from .render import mask
        clean = mask(text, tox.spans) if self.mask_for_intent else text
        intent, isc = self.intent_clf(clean or text)
        t["intent"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        # slots are extracted from the *masked* text: a verbatim slot must never carry abuse
        # (bug found in the live demo: prior_contact = "the fourth fucking time")
        sl = slotmod.extract(mask(text, tox.spans) if self.mask_slots else text)
        t["slots"] = (time.perf_counter() - t2) * 1000
        d.intent, d.intent_score, d.slots = intent, isc, sl
        t3 = time.perf_counter()
        if hasattr(self.renderer, "last_score"):
            self.renderer.last_score = isc
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


    def _gemini_path(self, text, tox, d, t):
        """v15: one structured Gemini call (intent + verbatim slots + calm message), verified; v11 ladder as fallback."""
        from .understand import TO_REGEX
        from .render import mask
        t1 = time.perf_counter()
        u = self.understanding.analyze(text, self.context)
        t["understand"] = (time.perf_counter() - t1) * 1000
        regex_slots = slotmod.extract(mask(text, tox.spans))
        if u is None:
            return self._fallback(text, tox, d, t)
        d.debug["understanding"] = u
        d.intent = u.get("intent", UNKNOWN)
        d.intent_score = float(u.get("intent_confidence") or 0)
        d.slots = {TO_REGEX.get(k, k): [v] for k, v in u["slots"].items()} or regex_slots
        if u.get("threat") or u.get("self_harm"):
            # union with the detector: Gemini catches conditional / implied threats the lexicon and Jigsaw models miss
            d.route = "ESCALATE"
            tox.is_threat = tox.is_toxic = True
            d.abuse_types = sorted(set(d.abuse_types) | {"self_harm" if u.get("self_harm") and not u.get("threat") else "threat"})
        if d.route == "ESCALATE":
            # threat calls get the complaint header + verbatim slots only — never a paraphrase of the threat sentence
            d.delivered = ALERT + self.renderer.fallback(text, d.intent if d.intent != UNKNOWN else UNKNOWN, regex_slots, tox) \
                if hasattr(self.renderer, "fallback") else ALERT + self._ladder(text, d.intent, regex_slots, tox)
            d.tier = 1
            d.timings_ms = t
            return d
        if d.route == "PASS" and self.gemini_detects and u.get("abusive_phrases"):
            # v16: Gemini-only detection (passive-aggressive belittling has no lexical or Jigsaw signal)
            d.route = "CALM"
            d.abuse_types = sorted(set(d.abuse_types) | {"belittling"})
            from .schema import Span
            for ph in u["abusive_phrases"]:
                i = text.lower().find(ph.lower())
                if i >= 0:
                    tox.spans.append(Span(i, i + len(ph), "belittling", ph))
            tox.is_toxic = True
        if d.route == "PASS":
            d.timings_ms = t
            return d
        if u.get("nothing_to_relay") and not tox.is_threat:
            d.delivered, d.tier = "", 1
            d.timings_ms = t
            return d
        out = (u.get("calm_message") or "").strip()
        reason = self.checker.check(text, out, regex_slots) if self.checker else None
        if reason:
            d.debug["rejected"] = reason
            if self.checker is not None and self.checker.withhold_if_empty(text, regex_slots, tox):
                d.delivered, d.tier = "", 3      # pure abuse: withhold rather than template it
                d.timings_ms = t
                return d
            return self._fallback(text, tox, d, t)
        from .pipeline import ALERT as _A
        d.delivered = (_A if d.route == "ESCALATE" else "") + out
        d.tier = 1
        d.timings_ms = t
        return d

    def _ladder(self, text, intent, slots, tox):
        r = self.renderer
        if hasattr(r, "last_score"):
            r.last_score = 1.0 if intent != UNKNOWN else 0.0
        return r(text, intent, slots, tox)

    def _fallback(self, text, tox, d, t):
        from .render import mask
        sl = slotmod.extract(mask(text, tox.spans))
        intent, isc = self.intent_clf(mask(text, tox.spans) or text)
        if not d.slots:
            d.slots = sl
        if d.intent == UNKNOWN:
            d.intent, d.intent_score = intent, isc
        if d.route == "PASS":
            d.timings_ms = t
            return d
        if hasattr(self.renderer, "last_score"):
            self.renderer.last_score = isc
        d.tier = 3
        d.delivered = (ALERT if d.route == "ESCALATE" else "") + self.renderer(text, intent if intent != UNKNOWN else UNKNOWN, sl, tox)
        d.timings_ms = t
        return d
