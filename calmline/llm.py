"""Tier-2 generative rewriters with deterministic verification.

GeminiRewriter  — Google Gemini via google-genai (Developer API key or Vertex AI ADC).
GemmaRewriter   — Gemma 3 run locally (open Google model; offline fallback / no API key needed).
Both are wrapped by VerifiedRewriter, which *never trusts* the generation:
  1. the output must not be abusive (re-scored by the CalmLine detector)
  2. every number in the output must appear in the caller's words (no invented addresses/times)
  3. every extracted slot value must survive
If any check fails, it falls back to the extractive renderer (tier 3)."""
from __future__ import annotations
import os
import re
import time
from functools import lru_cache

from .textnorm import fact_present, numbers

PROMPT_V2 = (
    "You relay a caller's message to a New York City 311 agent, speaking in the caller's place.\n"
    "Turn the caller's words into what a calm, polite caller would have said: first person, natural spoken English, "
    "short sentences, warm but plain.\n"
    "Keep EVERY fact: addresses, apartment numbers, streets, times, durations, counts, reference numbers, vehicles, "
    "people affected, previous calls, and exactly what the caller is asking for.\n"
    "Drop all insults, profanity, sexual remarks, slurs, threats and rhetorical jabs. Do not mention that anything was removed.\n"
    "Never add a fact, request or detail the caller did not say. If the words carry no request, do not invent one.\n"
    "If nothing meaningful is left once the abuse is removed (e.g. the words are only an insult), output exactly: [NOTHING]\n"
    "Output only the message.\n\nCaller's words: {text}\n\nCalm message:"
)

PROMPT = (
    "You relay a caller's message to a New York City 311 agent who must not be exposed to abuse.\n"
    "Rewrite the caller's message as a calm, polite first-person message to the agent.\n"
    "Rules:\n"
    "- Keep EVERY factual detail: addresses, apartment numbers, street names, times, durations, counts, reference numbers, "
    "vehicles, people affected, previous calls, and what the caller is asking for.\n"
    "- Remove all insults, profanity, sexual remarks, slurs and threats completely. Do not mention that they were removed.\n"
    "- Do not add any information that is not in the caller's message.\n"
    "- Output only the rewritten message, nothing else.\n\n"
    "Caller's message: {text}\n\nRewritten message:"
)


class GeminiRewriter:
    name = "gemini"

    def __init__(self, model=None, prompt=None, cached=True):
        from . import gemini as G
        self.G = G
        self.models = [model] if model else G.TEXT_MODELS
        self.model = self.models[0]
        self.prompt = prompt or PROMPT_V2
        self.cached = cached
        self.last_model = None
        self.context = ""   # sentences already relayed in this call (sentence-level relay)
        self.client = G.client()
        from google.genai import types
        # This is a content-moderation use case: the input is abusive by construction.
        self.cfg = types.GenerateContentConfig(
            temperature=0.0, max_output_tokens=256,
            safety_settings=[types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in (
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")])

    def generate(self, text):
        prompt = self.prompt.format(text=text)
        if self.context:
            prompt = (f"Already relayed earlier in this call (for context only, do not repeat it or greet again): {self.context}\n\n" + prompt)
        out, self.last_model = self.G.generate_text(prompt, config=self.cfg, models=self.models, use_cache=self.cached)
        return out.strip().strip('"')


@lru_cache(maxsize=2)
def _load_gemma(model_id, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().to(device)
    return tok, mdl


class GemmaRewriter:
    def __init__(self, model_id="unsloth/gemma-3-1b-it", device=None, max_new_tokens=160):
        import torch
        self.model_id, self.max_new = model_id, max_new_tokens
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.name = "gemma(" + model_id.split("/")[-1] + ")"

    def generate(self, text):
        import torch
        tok, mdl = _load_gemma(self.model_id, self.device)
        msgs = [{"role": "user", "content": PROMPT.format(text=text)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            out = mdl.generate(**ids, max_new_tokens=self.max_new, do_sample=False)
        s = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return s.split("\n\n")[0].strip().strip('"')


ALLOWED_NOVEL = set("""hello hi calling call report reporting would like please could kindly need needs needed thank thanks
regarding about concerned concern issue problem situation there here also currently still assist assistance help request requesting
want wanted ask asking someone somebody send sent check look looked address located location apartment street avenue i'm i've i'd
it's that's there's has been being am are was were is be let know you your sir madam okay sorry""".split())


class VerifiedRewriter:
    """Callable with the renderer signature (text, intent, slots, tox) -> str."""

    def __init__(self, gen, detector, fallback, mask_input=True):
        self.gen, self.detector, self.fallback, self.mask_input = gen, detector, fallback, mask_input
        self.name = f"verified({gen.name})"
        self.stats = {"calls": 0, "accepted": 0, "rej_toxic": 0, "rej_halluc": 0, "rej_facts": 0, "rej_error": 0, "rej_unsupported": 0, "ms": []}

    def __call__(self, text, intent, slots, tox):
        from .render import mask
        self.stats["calls"] += 1
        src = mask(text, [s for s in tox.spans if s.label not in ("threat", "self_harm")]) if self.mask_input else text
        t0 = time.perf_counter()
        try:
            out = self.gen.generate(src)
        except Exception as e:  # network / quota / safety block -> deterministic fallback
            self.stats["rej_error"] += 1
            return self.fallback(text, intent, slots, tox)
        finally:
            self.stats["ms"].append((time.perf_counter() - t0) * 1000)
        if out.strip().upper().strip(".") in ("[NOTHING]", "NOTHING"):
            self.stats["accepted"] += 1
            return ""          # withheld: pure abuse, nothing to relay
        reason = self.check(text, out, slots)
        if reason:
            self.stats["rej_" + reason] += 1
            if self.withhold_if_empty(text, slots, tox):
                return ""
            return self.fallback(text, intent, slots, tox)
        self.stats["accepted"] += 1
        return out

    @staticmethod
    def withhold_if_empty(text, slots, tox):
        """Sentence-level relay: a sentence with no facts left after removing the abuse is not relayed at all."""
        from .render import mask
        from .textnorm import content_tokens
        rest = mask(text, tox.spans)
        facty = any(slots.get(k) for k in ("address", "street", "cross_street", "apartment", "reference", "plate", "vehicle", "time", "duration", "request"))
        return not facty and len(content_tokens(rest)) < 5

    def check(self, text, out, slots):
        if not out or len(out) < 8:
            return "error"
        v = self.detector(out)
        if v.is_toxic or v.is_threat:
            return "toxic"
        if numbers(out) - numbers(text) - {"311"}:
            return "halluc"
        # unsupported claims: content words in the output that the caller never said (Gemma 1B invented
        # "I've called before", "123 Maple Street", "I'd like to know when..." in our first probe)
        from .textnorm import content_tokens
        src = set(content_tokens(text))
        novel = [w for w in content_tokens(out) if w not in src and w not in ALLOWED_NOVEL]
        if len(novel) > max(3, 0.25 * len(content_tokens(out))):
            return "unsupported"
        keep = [v for k in ("address", "apartment", "reference", "plate", "cross_street", "street") for v in slots.get(k, [])]
        if any(not fact_present(re.sub(r"^(?:at|on|in|near|corner of)\s+", "", v, flags=re.I), out) for v in keep):
            return "facts"
        return None
