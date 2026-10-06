# CalmLine — experiment results

All numbers are produced by `python -m eval.run_text <versions>`; every version is a frozen config in `calmline/versions.py`.

## Text pipeline — held-out hand-written TEST set (n=134, frozen before any tuning)

| ver | protected ↑ | leak ↓ | abuse recall ↑ | threat recall ↑ | threat prec ↑ | FPR test ↓ | FPR hard-neg ↓ | sufficiency ↑ | fact recall ↑ | wrong issue ↓ | halluc ↓ | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 87.9 | 12.0 | 92.8 | 100.0 | 92.9 | 2.1 | 1.0 | 44.6 | 71.5 | 29.5 | 0.0 | 0 |
| v1 | 92.8 | 7.2 | 98.8 | 100.0 | 81.2 | 12.8 | 10.2 | 38.6 | 68.7 | 12.1 | 0.0 | 31 |
| v2 | 78.3 | 21.7 | 98.8 | 100.0 | 81.2 | 12.8 | 10.2 | 81.9 | 90.3 | 12.1 | 0.0 | — |
| v3 | 84.3 | 15.7 | 98.8 | 100.0 | 92.9 | 4.3 | 1.0 | 85.5 | 92.3 | 12.9 | 0.0 | — |
| v4 | 91.6 | 8.4 | 98.8 | 100.0 | 92.9 | 4.3 | 4.1 | 81.9 | 91.0 | 11.8 | 0.0 | 57 |
| v5 | 91.6 | 8.4 | 97.6 | 100.0 | 92.9 | 8.5 | 4.1 | 81.9 | 91.4 | 11.4 | 0.0 | — |
| v6 | 91.6 | 8.4 | 97.6 | 100.0 | 86.7 | 8.5 | 4.1 | 81.9 | 91.4 | 11.4 | 0.0 | — |
| v7 | 91.6 | 8.4 | 97.6 | 100.0 | 92.9 | 8.5 | 3.1 | 81.9 | 91.4 | 11.4 | 0.0 | — |
| v8 | 91.6 | 8.4 | 97.6 | 100.0 | 92.9 | 8.5 | 1.0 | 81.9 | 91.4 | 11.4 | 0.0 | 51 |
| v9 | 91.6 | 8.4 | 97.6 | 100.0 | 92.9 | 8.5 | 1.0 | 84.3 | 91.6 | 2.3 | 0.0 | — |
| v10 | 92.8 | 7.2 | 97.6 | 100.0 | 92.9 | 8.5 | 1.0 | 83.1 | 90.4 | 2.3 | 0.0 | 48 |
| v11 † | 98.8 | 1.2 | 100.0 | 100.0 | 92.9 | 2.1 | 1.0 | 78.3 | 85.8 | 3.5 | 0.0 | 50 |
| v12 † | 98.8 | 1.2 | 100.0 | 100.0 | 92.9 | 2.1 | — | 78.3 | 85.8 | 3.5 | 0.0 | — |

*protected* = abusive utterance detected **and** no abusive span reached the agent. *sufficiency* = every gold fact needed to resolve the complaint survives in what the agent receives (abusive utterances only). *FPR* = clean utterances that were rewritten. *halluc* = numbers in the agent's text that the caller never said. *wrong issue* = the agent's text asserts the wrong complaint type. *p95 ms* = clean per-utterance latency on Apple M5 (`eval/latency.py`), measured for the milestone versions. † = post-hoc: built after reading v10's test errors, so its test numbers are optimistic.

## LLM tier ablation (local Gemma 3 1B; Gemini needs an API key — see docs/DEPLOY.md)

| config | split | protected | sufficiency | fact recall | halluc | LLM accepted / calls | reject reasons | LLM ms p50 |
|---|---|---|---|---|---|---|---|---|
| v12 | dev | 100.0 | 77.3 | 85.8 | 0.0 | 0 / 3 | halluc 2, unsupported 1 | 2046 |
| v12 | test | 98.8 | 78.3 | 85.8 | 0.0 | 2 / 10 | halluc 5, unsupported 3 | 1959 |
| llm_all | dev | 100.0 | 77.3 | 88.0 | 0.0 | 8 / 23 | toxic 1, halluc 6, unsupported 8 | 1478 |
| llm_all | test | 97.6 | 74.7 | 86.4 | 0.0 | 43 / 110 | toxic 2, halluc 23, facts 2, unsupported 40 | 1193 |
| llm_raw | dev | 95.5 | 77.3 | 89.3 | 15.0 | — / — | unverified | — |
| llm_raw | test | 91.6 | 74.7 | 90.0 | 13.4 | — / — | unverified | — |

## Same metrics on development data (used for tuning)

| ver | split | protected | leak | threat recall | FPR | sufficiency |
|---|---|---|---|---|---|---|
| v0 | dev | 59.1 | 40.9 | 100.0 | 0.0 | 54.5 |
| v0 | synth | 76.7 | 21.5 | 73.7 | 0.0 | 37.6 |
| v1 | dev | 86.4 | 13.6 | 100.0 | 11.1 | 45.5 |
| v1 | synth | 84.5 | 13.7 | 80.7 | 3.3 | 33.2 |
| v2 | dev | 81.8 | 18.2 | 100.0 | 11.1 | 72.7 |
| v2 | synth | 64.2 | 33.9 | 80.7 | 3.3 | 87.8 |
| v3 | dev | 95.5 | 4.5 | 100.0 | 0.0 | 77.3 |
| v3 | synth | 66.8 | 31.4 | 73.7 | 0.0 | 86.3 |
| v4 | dev | 100.0 | 0.0 | 100.0 | 0.0 | 77.3 |
| v4 | synth | 98.2 | 1.8 | 100.0 | 0.0 | 83.9 |
| v5 | dev | 100.0 | 0.0 | 100.0 | 11.1 | 72.7 |
| v5 | synth | 97.4 | 2.6 | 100.0 | 0.0 | 84.5 |
| v6 | dev | 100.0 | 0.0 | 100.0 | 11.1 | 72.7 |
| v6 | synth | 97.4 | 2.6 | 100.0 | 0.0 | 83.7 |
| v7 | dev | 100.0 | 0.0 | 100.0 | 11.1 | 72.7 |
| v7 | synth | 97.4 | 2.6 | 100.0 | 0.0 | 84.5 |
| v8 | dev | 100.0 | 0.0 | 100.0 | 5.6 | 72.7 |
| v8 | synth | 97.4 | 2.6 | 100.0 | 0.0 | 84.5 |
| v9 | dev | 100.0 | 0.0 | 100.0 | 5.6 | 77.3 |
| v9 | synth | 97.4 | 2.6 | 100.0 | 0.0 | 88.3 |
| v10 | dev | 100.0 | 0.0 | 100.0 | 5.6 | 81.8 |
| v10 | synth | 98.2 | 1.8 | 100.0 | 0.0 | 88.3 |
| v11 | dev | 100.0 | 0.0 | 100.0 | 5.6 | 77.3 |
| v11 | synth | 97.7 | 2.3 | 100.0 | 0.0 | 81.6 |
| v12 | dev | 100.0 | 0.0 | 100.0 | 5.6 | 77.3 |

## Version log

- **v0** — Baseline: regex lexicon + keyword intent + pure template (the original 'pre-written phrases' idea)
- **v1** — Detection: lexicon ∪ toxic-bert (Jigsaw, clause-level) · Intent: bge-small nearest-prototype (open-set τ=0.6) · pure template
- **v2** — v1 + hybrid renderer: template header for tone + extractive detail clauses with abusive spans removed (condBERT-style)
- **v3** — v2 + directedness gate on model-only flags (Waseem et al. 2017) + render-then-verify detail clauses + threats never echo wording + generic header for unknown intents
- **v4** — v3 + residue-free masking: abusive clauses with no slot content are dropped, mild profanity stripped in rewrites, generalised threat/self-harm/identity lexicon (synthetic-dev errors)
- **v5** — v4 with detector backbone swapped toxic-bert -> unbiased-toxic-roberta (better threat recall on Civil Comments: 0.29 -> 0.73)
- **v6** — v5 + zero-shot NLI safety net for threats/self-harm (DeBERTa-v3 zeroshot, entailment >= 0.9)
- **v7** — Detector: lexicon(v7) ∪ unbiased-toxic-roberta with directedness gate, but threats bypass the gate (third-party threats escalate); self-harm needs a first-person subject
- **v8** — v7 + context filter: model-only flags dropped in figurative ('killing me') or concern ('worried someone could die') clauses without first-person intent
- **v9** — v8 + confidence-gated header: the issue type is only asserted in the agent's text when intent similarity >= 0.70 (92.5% precision on dev), otherwise a neutral header + the caller's cleaned words
- **v10** — v9 + slots extracted from masked text (a verbatim slot can never carry abuse — bug found in live demo)
- **v11** — POST-HOC (after reading v10 TEST errors): topic-word neutralisation before model scoring (Dixon et al. 2018), threat calls get header+slots only, 4 lexicon fixes. Test numbers for v11+ are contaminated.
- **v12** — v11 + tier 2: verified LLM rewrite (Gemini if GEMINI_API_KEY is set, else local Gemma 3 1B) for utterances whose intent is unknown

## Detector benchmark (utterance level)

| set | detector | abuse P | abuse R | abuse F1 | threat R | threat AUC | ms/utt |
|---|---|---|---|---|---|---|---|
| dev | lexicon_v0 | 1.000 | 0.727 | 0.842 | 1.000 | — | 0.1 |
| dev | lexicon_v4 | 1.000 | 0.955 | 0.977 | 1.000 | — | 0.1 |
| dev | toxic-bert | 0.889 | 0.727 | 0.800 | 0.250 | 0.764 | 12.0 |
| dev | unbiased-roberta | 0.889 | 0.727 | 0.800 | 0.500 | 0.722 | 11.1 |
| dev | hybrid_v4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 10.7 |
| dev | nli | 0.923 | 0.545 | 0.686 | 0.500 | 1.000 | 55.6 |
| synth | lexicon_v0 | 1.000 | 0.839 | 0.913 | 0.737 | — | 0.1 |
| synth | lexicon_v4 | 1.000 | 0.969 | 0.984 | 1.000 | — | 0.1 |
| synth | toxic-bert | 0.973 | 0.661 | 0.787 | 0.175 | 0.847 | 13.0 |
| synth | unbiased-roberta | 0.971 | 0.686 | 0.804 | 0.210 | 0.706 | 12.7 |
| synth | hybrid_v4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 13.1 |
| synth | nli | 1.000 | 0.471 | 0.641 | 0.632 | 0.990 | 30.2 |
| civil | lexicon_v0 | 0.953 | 0.217 | 0.353 | 0.118 | — | 0.2 |
| civil | lexicon_v4 | 0.951 | 0.229 | 0.369 | 0.127 | — | 0.3 |
| civil | toxic-bert | 0.952 | 0.698 | 0.805 | 0.294 | 0.919 | 44.9 |
| civil | unbiased-roberta | 0.952 | 0.878 | 0.913 | 0.729 | 0.965 | 53.6 |
| civil | hybrid_v4 | 0.952 | 0.476 | 0.635 | 0.190 | 0.916 | 54.5 |
| civil | nli | 0.845 | 0.665 | 0.745 | 0.394 | 0.910 | 117.0 |
| dev | hybrid_v4_tb | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 12.3 |
| dev | hybrid_unb_tb | 0.917 | 1.000 | 0.957 | 1.000 | 1.000 | 10.6 |
| synth | hybrid_v4_tb | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 14.1 |
| synth | hybrid_unb_tb | 1.000 | 0.995 | 0.997 | 1.000 | 1.000 | 13.6 |
| civil | hybrid_v4_tb | 0.950 | 0.489 | 0.646 | 0.348 | 0.916 | 56.0 |
| civil | hybrid_unb_tb | 0.962 | 0.740 | 0.836 | 0.724 | 0.957 | 58.2 |

## Acoustic aggression (CREMA-D, speaker-independent: 30 held-out speakers, n=722)

| features / model | AUC | recall | recall (medium+high anger) | precision |
|---|---|---|---|---|
| loudness_only/logreg | 0.868 | 0.754 | 0.900 | 0.430 |
| loudness_only+spknorm/logreg | 0.889 | 0.842 | 1.000 | 0.451 |
| prosody/logreg | 0.873 | 0.746 | 0.800 | 0.443 |
| prosody/gbm | 0.851 | 0.526 | 0.700 | 0.541 |
| prosody+spknorm/logreg | 0.909 | 0.860 | 0.950 | 0.483 |
| prosody+spknorm/gbm | 0.878 | 0.640 | 0.600 | 0.589 |
| prosody+mfcc+spknorm/logreg | 0.921 | 0.886 | 1.000 | 0.502 |
| prosody+mfcc+spknorm/gbm | 0.892 | 0.614 | 0.700 | 0.625 |

Look-ahead compressor: peak loudness reduced by **11.8 dB** on high-intensity anger vs **0.3 dB** on neutral speech.

## Real-time delay buffer — abusive-word leak vs. playback delay (TEST audio, TTS-rendered)

| STT | hop (s) | policy | RTF | leak@1s | leak@1.5s | leak@2s | leak@3s | leak@4s (accuracy ceiling) | false-muted s per clean utt @2s |
|---|---|---|---|---|---|---|---|---|---|
| base.en | 0.5 | committed | 1.12 | 94.5 | 66.9 | 49.2 | 40.6 | 37.0 | 0.08 |
| base.en | 0.5 | window | 1.18 | 64.2 | 41.7 | 31.9 | 26.0 | 25.6 | 0.03 |
| base.en | 0.5 | window+bias | 1.62 | 81.9 | 71.3 | 64.6 | 44.1 | 37.0 | 0.03 |
| small.en | 0.5 | window+bias | 3.20 | 100.0 | 90.9 | 73.2 | 46.1 | 24.4 | 0.07 |
| tiny.en | 0.25 | window+bias | 1.02 | 32.7 | 19.7 | 18.1 | 16.5 | 15.7 | 0.32 |
| tiny.en | 0.5 | window_edge | 0.38 | 59.8 | 25.6 | 20.5 | 19.3 | 18.1 | 0.13 |
| tiny.en | 0.5 | window | 0.39 | 37.4 | 21.7 | 18.1 | 17.7 | 16.9 | 0.18 |

RTF > 1 means the recogniser falls behind real time on this laptop CPU, so decisions arrive late and leak grows; the accuracy ceiling (leak@4s) separates recognition misses from lateness. The committed (LocalAgreement) row used an earlier, stricter mute-coverage criterion.
