"""One Gemini call per sentence: complaint type + slots + calm message, as structured JSON.

Trust rules (same principle as the template ladder: facts can only be *extracted*):
  * every slot value must literally appear in the caller's words, otherwise it is dropped and counted;
  * the calm message passes the VerifiedRewriter checks (not abusive, no new numbers, no unsupported claims);
  * the complaint type comes from a closed list (26 NYC 311 intents + UNKNOWN)."""
from __future__ import annotations
import json
import re

from .intents import INTENTS_ALL as INTENTS
from .textnorm import fact_present

SLOT_KEYS = ["location", "apartment", "when", "how_long", "previous_contact", "reference_number", "vehicle", "plate",
             "people_affected", "caller_request"]

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING", "enum": list(INTENTS) + ["UNKNOWN"]},
        "intent_confidence": {"type": "NUMBER"},
        "slots": {"type": "OBJECT", "properties": {k: {"type": "STRING"} for k in SLOT_KEYS}},
        "abusive_phrases": {"type": "ARRAY", "items": {"type": "STRING"}},
        "calm_message": {"type": "STRING"},
        "nothing_to_relay": {"type": "BOOLEAN"},
        "threat": {"type": "BOOLEAN"},
        "self_harm": {"type": "BOOLEAN"},
    },
    "required": ["intent", "intent_confidence", "slots", "abusive_phrases", "calm_message", "nothing_to_relay", "threat", "self_harm"],
}

CATALOG = "\n".join(f"- {k}: {v['issue']} (NYC 311: {', '.join(v['nyc'])})" for k, v in INTENTS.items())

PROMPT = """You are CalmLine, a relay that protects New York City 311 call agents from abusive callers.
Analyse the caller's latest words (one sentence or two) and return JSON.

intent: the complaint type of the call, from this list (UNKNOWN if none fits or it cannot be told yet):
{catalog}
Use the earlier sentences of the call for context when the latest words alone are ambiguous.

slots: copy each value VERBATIM from the latest words (an exact substring); leave a slot out if it is not said.
  location (address, street, cross streets), apartment, when (start time/day/frequency), how_long, previous_contact,
  reference_number, vehicle, plate, people_affected (vulnerable people, e.g. "three kids", "my mother"), caller_request.
abusive_phrases: exact substrings that are insults, profanity, sexual remarks, slurs or threats — and also polite-sounding
  belittling of the agent: sarcasm or condescension about the agent's intelligence, competence or job, and patronising pet
  names. Judge whether the phrase is aimed at the agent as a person. Do NOT include the caller's own feelings, urgency or
  legitimate complaints about the service (frustration, repeated calls, waiting time, nobody coming): those are information
  the agent needs.
calm_message: what a calm, polite caller would have said instead — first person, natural spoken English, every fact kept,
  all abuse dropped, nothing added (no invented requests, numbers or details). Questions stay questions.
  When an insult stands for a person ("some asshole parked..."), say "someone"; when it describes the agent or agency, drop it.
nothing_to_relay: true if the latest words are only abuse with no information for the agent.
threat: true if the caller threatens violence or harm to the agent, staff, an office, or anyone else — including conditional threats
  ("if nobody comes I'll come down there and ...") and implied ones ("I know where your office is"). Fear or worry that someone
  could get hurt is NOT a threat.
self_harm: true if the caller says they may hurt or kill themselves.

Earlier in this call: {context}
Latest words: {text}"""


class GeminiUnderstanding:
    name = "gemini-understand"

    def __init__(self):
        from . import gemini as G
        from google.genai import types
        self.G = G
        self.cfg = types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json", response_schema=SCHEMA, max_output_tokens=600,
            safety_settings=[types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in (
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")])
        self.stats = {"calls": 0, "slots_kept": 0, "slots_dropped_not_in_source": 0, "errors": 0}

    def analyze(self, text: str, context: str = "") -> dict | None:
        self.stats["calls"] += 1
        try:
            raw, model = self.G.generate_text(PROMPT.format(catalog=CATALOG, context=context or "(start of call)", text=text), config=self.cfg)
            out = json.loads(raw)
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1
            return None
        slots, dropped = {}, []
        for k, v in (out.get("slots") or {}).items():
            v = (v or "").strip()
            if not v or k not in SLOT_KEYS:
                continue
            if v.lower() in text.lower() or fact_present(v, text):
                slots[k] = v; self.stats["slots_kept"] += 1
            else:
                dropped.append((k, v)); self.stats["slots_dropped_not_in_source"] += 1
        out["slots"], out["slots_dropped"], out["model"] = slots, dropped, model
        out["abusive_phrases"] = [p for p in out.get("abusive_phrases", []) if p and p.lower() in text.lower()]
        return out


# map Gemini slot names onto the regex slot names used by the template renderer / UI
TO_REGEX = {"location": "address", "apartment": "apartment", "when": "time", "how_long": "duration", "previous_contact": "prior_contact",
            "reference_number": "reference", "vehicle": "vehicle", "plate": "plate", "people_affected": "vulnerable", "caller_request": "request"}
