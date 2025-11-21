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
    tone: str  # sentiment label: positive/negative/neutral
    voice: str
    persona: str
    escalation: bool = False
    notes: str = ""
    style_hint: str = "balanced"

    def apply(self, message: str) -> str:
        """Adjust the outgoing text to match tone and voice guidance."""
        segments = []
        prefix = self._prefix()
        suffix = self._suffix()
        if prefix:
            segments.append(prefix)
        segments.append(message)
        if suffix:
            segments.append(suffix)
        final = " ".join(segment.strip() for segment in segments if segment)
        return final.strip()

    def _prefix(self) -> str:
        if self.escalation:
            return "I know this is urgent, and I'm here with you."
        if self.style_hint == "empathetic":
            return "I appreciate you sharing how you're feeling."
        if self.style_hint == "encouraging":
            return "Love the momentum!"
        return ""

    def _suffix(self) -> str:
        if self.escalation:
            return "I'm routing this for immediate follow-up while keeping you updated."
        if self.style_hint == "encouraging":
            return "Let's keep things moving!"
        if self.style_hint == "empathetic":
            return "We'll tackle this together step by step."
        return "Let me know if you'd like me to dive deeper."


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
            tone=sentiment.label if sentiment else "neutral",
            voice=tone.voice if tone else "informative",
            persona=persona,
            escalation=tone.escalate if tone else False,
            notes=tone.notes if tone else "",
            style_hint=tone.style if tone else "balanced",
        )
        return cls(request=request, guidance=guide)


