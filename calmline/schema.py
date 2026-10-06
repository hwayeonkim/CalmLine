from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

ROUTES = ("PASS", "CALM", "ESCALATE")


@dataclass
class Span:
    start: int
    end: int
    label: str          # profanity | insult | sexual | identity | threat | self_harm
    text: str
    score: float = 1.0


@dataclass
class ToxicityResult:
    is_toxic: bool
    is_threat: bool
    score: float                      # max toxicity in [0,1]
    threat_score: float
    spans: list[Span] = field(default_factory=list)
    labels: dict = field(default_factory=dict)


@dataclass
class Decision:
    route: str                        # PASS | CALM | ESCALATE
    delivered: str                    # what the agent sees / hears as text
    intent: str = "UNKNOWN"
    intent_score: float = 0.0
    slots: dict = field(default_factory=dict)
    tier: int = 0                     # 0 pass, 1 template, 2 llm rewrite, 3 masked transcript
    abuse_types: list = field(default_factory=list)
    anger: float = 0.0                # 0..1 gauge shown to agent
    timings_ms: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
