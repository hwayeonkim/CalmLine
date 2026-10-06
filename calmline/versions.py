"""Version registry — each version is a frozen configuration so results stay comparable."""
from .pipeline import Pipeline
from .toxicity import LexiconDetector, ModelDetector, HybridDetector
from .intent_clf import KeywordIntent, EmbeddingIntent
from .render import TemplateRenderer, HybridRenderer, MaskRenderer

DESCRIPTIONS = {
    "v0": "Baseline: regex lexicon + keyword intent + pure template (the original 'pre-written phrases' idea)",
    "v1": "Detection: lexicon ∪ toxic-bert (Jigsaw, clause-level) · Intent: bge-small nearest-prototype (open-set τ=0.6) · pure template",
    "v2": "v1 + hybrid renderer: template header for tone + extractive detail clauses with abusive spans removed (condBERT-style)",
    "v3": "v2 + directedness gate on model-only flags (Waseem et al. 2017) + render-then-verify detail clauses + threats never echo wording + generic header for unknown intents",
}


def build(version: str) -> Pipeline:
    if version == "v0":
        return Pipeline(LexiconDetector(extended=False), KeywordIntent(), TemplateRenderer(), name="v0")
    if version == "v1":
        return Pipeline(HybridDetector(ModelDetector("unitary/toxic-bert"), LexiconDetector(extended=False)), EmbeddingIntent(threshold=0.6), TemplateRenderer(), name="v1")
    if version == "v2":
        return Pipeline(HybridDetector(ModelDetector("unitary/toxic-bert"), LexiconDetector(extended=False)), EmbeddingIntent(threshold=0.6), HybridRenderer(), name="v2")
    if version == "v3":
        det = HybridDetector(ModelDetector("unitary/toxic-bert"), directed_gate=True)
        return Pipeline(det, EmbeddingIntent(threshold=0.6), HybridRenderer(verify=det), name="v3", unknown_mode="generic")
    raise KeyError(version)
