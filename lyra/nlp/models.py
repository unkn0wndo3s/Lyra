"""Dataclasses describing Lyra's NLP artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenData:
    text: str
    lemma: str
    pos: str
    tag: str
    dep: str


@dataclass
class EntityData:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class SentimentResult:
    polarity: float
    subjectivity: float
    label: str


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NLPResult:
    text: str
    tokens: List[TokenData]
    entities: List[EntityData]
    sentiment: SentimentResult
    intent: IntentPrediction
    doc: Any = field(repr=False, default=None)


@dataclass
class DialogueTurn:
    speaker: str
    text: str
    intent: str
    sentiment: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueContext:
    session_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def append(self, turn: DialogueTurn, *, max_turns: Optional[int] = None) -> None:
        self.turns.append(turn)
        if max_turns and len(self.turns) > max_turns:
            excess = len(self.turns) - max_turns
            del self.turns[0:excess]


@dataclass
class DialogueResponse:
    session_id: str
    nlp: NLPResult
    context: DialogueContext


