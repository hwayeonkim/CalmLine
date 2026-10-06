"""Version registry — each version is a frozen configuration so results stay comparable."""
from .pipeline import Pipeline
from .toxicity import LexiconDetector, ModelDetector, HybridDetector
from .intent_clf import KeywordIntent, EmbeddingIntent
from .render import TemplateRenderer, HybridRenderer, MaskRenderer, ConfidenceGatedRenderer

BEST = "v10"   # updated as experiments conclude

DESCRIPTIONS = {
    "v0": "Baseline: regex lexicon + keyword intent + pure template (the original 'pre-written phrases' idea)",
    "v1": "Detection: lexicon ∪ toxic-bert (Jigsaw, clause-level) · Intent: bge-small nearest-prototype (open-set τ=0.6) · pure template",
    "v2": "v1 + hybrid renderer: template header for tone + extractive detail clauses with abusive spans removed (condBERT-style)",
    "v3": "v2 + directedness gate on model-only flags (Waseem et al. 2017) + render-then-verify detail clauses + threats never echo wording + generic header for unknown intents",
    "v4": "v3 + residue-free masking: abusive clauses with no slot content are dropped, mild profanity stripped in rewrites, generalised threat/self-harm/identity lexicon (synthetic-dev errors)",
    "v5": "v4 with detector backbone swapped toxic-bert -> unbiased-toxic-roberta (better threat recall on Civil Comments: 0.29 -> 0.73)",
    "v6": "v5 + zero-shot NLI safety net for threats/self-harm (DeBERTa-v3 zeroshot, entailment >= 0.9)",
    "v7": "Detector: lexicon(v7) ∪ unbiased-toxic-roberta with directedness gate, but threats bypass the gate (third-party threats escalate); self-harm needs a first-person subject",
    "v8": "v7 + context filter: model-only flags dropped in figurative ('killing me') or concern ('worried someone could die') clauses without first-person intent",
    "v9": "v8 + confidence-gated header: the issue type is only asserted in the agent's text when intent similarity >= 0.70 (92.5% precision on dev), otherwise a neutral header + the caller's cleaned words",
    "v10": "v9 + slots extracted from masked text (a verbatim slot can never carry abuse — bug found in live demo)",
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
    if version == "v4":
        det = HybridDetector(ModelDetector("unitary/toxic-bert"), LexiconDetector(extended="v4"), directed_gate=True)
        return Pipeline(det, EmbeddingIntent(threshold=0.6), HybridRenderer(verify=det, drop_residue=True), name="v4",
                        unknown_mode="generic", mask_mild=True)
    if version in ("v5", "v6"):
        det = HybridDetector(ModelDetector("unitary/unbiased-toxic-roberta"), LexiconDetector(extended="v4"), directed_gate=True)
        if version == "v6":
            from .toxicity import SafetyNetDetector
            det = SafetyNetDetector(det)
        return Pipeline(det, EmbeddingIntent(threshold=0.6), HybridRenderer(verify=det, drop_residue=True), name=version,
                        unknown_mode="generic", mask_mild=True)
    if version == "v7":
        det = HybridDetector(ModelDetector("unitary/unbiased-toxic-roberta"), LexiconDetector(extended="v7"), directed_gate=True, threat_bypass=True)
        return Pipeline(det, EmbeddingIntent(threshold=0.6), HybridRenderer(verify=det, drop_residue=True), name=version,
                        unknown_mode="generic", mask_mild=True)
    if version == "v8":
        det = HybridDetector(ModelDetector("unitary/unbiased-toxic-roberta"), LexiconDetector(extended="v7"), directed_gate=True,
                             threat_bypass=True, context_filter=True)
        return Pipeline(det, EmbeddingIntent(threshold=0.6), HybridRenderer(verify=det, drop_residue=True), name=version,
                        unknown_mode="generic", mask_mild=True)
    if version == "v9":
        det = HybridDetector(ModelDetector("unitary/unbiased-toxic-roberta"), LexiconDetector(extended="v7"), directed_gate=True,
                             threat_bypass=True, context_filter=True)
        return Pipeline(det, EmbeddingIntent(threshold=0.6), ConfidenceGatedRenderer(assert_th=0.7, verify=det, drop_residue=True), name=version,
                        unknown_mode="generic", mask_mild=True)
    if version == "v10":
        det = HybridDetector(ModelDetector("unitary/unbiased-toxic-roberta"), LexiconDetector(extended="v7"), directed_gate=True,
                             threat_bypass=True, context_filter=True)
        return Pipeline(det, EmbeddingIntent(threshold=0.6), ConfidenceGatedRenderer(assert_th=0.7, verify=det, drop_residue=True),
                        name=version, unknown_mode="generic", mask_mild=True, mask_slots=True)
    raise KeyError(version)
