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

    def __init__(self, model=None):
        from google import genai
        self.model = model or os.getenv("CALMLINE_GEMINI_MODEL", "gemini-2.5-flash-lite")
        if os.getenv("USE_VERTEX") == "1":
            self.client = genai.Client(vertexai=True, project=os.environ["GCP_PROJECT"], location=os.getenv("GCP_LOCATION", "global"))
        else:
            self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        from google.genai import types
        # This is a content-moderation use case: the input is abusive by construction.
        self.cfg = types.GenerateContentConfig(
            temperature=0.0, max_output_tokens=256,
            safety_settings=[types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in (
                "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")])

    def generate(self, text):
        r = self.client.models.generate_content(model=self.model, contents=PROMPT.format(text=text), config=self.cfg)
        return (r.text or "").strip()


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


class VerifiedRewriter:
    """Callable with the renderer signature (text, intent, slots, tox) -> str."""

    def __init__(self, gen, detector, fallback, mask_input=True):
        self.gen, self.detector, self.fallback, self.mask_input = gen, detector, fallback, mask_input
        self.name = f"verified({gen.name})"
        self.stats = {"calls": 0, "accepted": 0, "rej_toxic": 0, "rej_halluc": 0, "rej_facts": 0, "rej_error": 0, "ms": []}

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
        reason = self.check(text, out, slots)
        if reason:
            self.stats["rej_" + reason] += 1
            return self.fallback(text, intent, slots, tox)
        self.stats["accepted"] += 1
        return out

    def check(self, text, out, slots):
        if not out or len(out) < 8:
            return "error"
        v = self.detector(out)
        if v.is_toxic or v.is_threat:
            return "toxic"
        if numbers(out) - numbers(text) - {"311"}:
            return "halluc"
        keep = [v for k in ("address", "apartment", "reference", "plate", "cross_street", "street") for v in slots.get(k, [])]
        if any(not fact_present(re.sub(r"^(?:at|on|in|near|corner of)\s+", "", v, flags=re.I), out) for v in keep):
            return "facts"
        return None
