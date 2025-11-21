"""Dataclasses for adaptive learning feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FeedbackRecord:
    session_id: str
    interaction_id: str
    input_text: str
    response_text: str
    reward: float
    intent: str
    sentiment: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PolicyStats:
    intent: str
    sample_count: int = 0
    cumulative_reward: float = 0.0
    average_reward: float = 0.0
    preferred_tone: Optional[str] = None
    preferred_voice: Optional[str] = None


@dataclass
class PolicyRecommendation:
    intent: str
    tone: str
    voice: str
    confidence: float
    notes: str = ""


@dataclass
class LearnedFact:
    key: str
    value: Any
    category: str = "learned_fact"
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


