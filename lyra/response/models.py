"""Dataclasses for Lyra's response generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lyra.nlp.models import DialogueContext, NLPResult, SentimentResult


@dataclass
class ResponseRequest:
    session_id: str
    text: str
    context: DialogueContext
    nlp: NLPResult
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseCandidate:
    text: str
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleGuide:
    tone: str
    voice: str
    persona: str
    escalation: bool = False
    notes: str = ""


@dataclass
class ResponsePlan:
    request: ResponseRequest
    guidance: StyleGuide
    selected_candidate: Optional[ResponseCandidate] = None
    alternatives: List[ResponseCandidate] = field(default_factory=list)

    @classmethod
    def from_request(
        cls,
        request: ResponseRequest,
        *,
        sentiment: Optional[SentimentResult] = None,
        persona: str = "assistant",
    ) -> "ResponsePlan":
        tone = sentiment.tone if sentiment else None
        guide = StyleGuide(
            tone=tone.style if tone else "neutral",
            voice=tone.voice if tone else "informative",
            persona=persona,
            escalation=tone.escalate if tone else False,
            notes=tone.notes if tone else "",
        )
        return cls(request=request, guidance=guide)


