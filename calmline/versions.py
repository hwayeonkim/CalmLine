"""Version registry — each version is a frozen configuration so results stay comparable."""
from .pipeline import Pipeline
from .toxicity import LexiconDetector, ModelDetector, HybridDetector
from .intent_clf import KeywordIntent, EmbeddingIntent
from .render import TemplateRenderer, HybridRenderer, MaskRenderer

DESCRIPTIONS = {
    "v0": "Baseline: regex lexicon + keyword intent + pure template (the original 'pre-written phrases' idea)",
}


def build(version: str) -> Pipeline:
    if version == "v0":
        return Pipeline(LexiconDetector(), KeywordIntent(), TemplateRenderer(), name="v0")
    raise KeyError(version)
