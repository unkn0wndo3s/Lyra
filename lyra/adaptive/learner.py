"""Adaptive learning services for Lyra."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional

from lyra.memory.manager import MemoryManager, MemoryScope

from .models import FeedbackRecord, PolicyRecommendation, PolicyStats


class AdaptiveLearner:
    """Simple reinforcement-style learner that records feedback per intent."""

    def __init__(
        self,
        *,
        memory_manager: Optional[MemoryManager] = None,
        max_history: int = 200,
    ) -> None:
        self.memory = memory_manager or MemoryManager()
        self.max_history = max_history

    # ------------------------------------------------------------------ #
    # Feedback ingestion
    # ------------------------------------------------------------------ #

    def record_feedback(self, record: FeedbackRecord) -> None:
        """Persist raw feedback and update aggregated stats."""
        self._persist_feedback(record)
        stats = self._load_stats(record.intent)
        stats.sample_count += 1
        stats.cumulative_reward += record.reward
        stats.average_reward = stats.cumulative_reward / max(1, stats.sample_count)
        tone_preference = record.metadata.get("tone") or record.metadata.get("style")
        voice_preference = record.metadata.get("voice")
        if tone_preference:
            stats.preferred_tone = tone_preference
        if voice_preference:
            stats.preferred_voice = voice_preference
        self._persist_stats(stats)

    # ------------------------------------------------------------------ #
    # Recommendations
    # ------------------------------------------------------------------ #

    def suggest_policy(
        self,
        intent: str,
        *,
        default_tone: str = "neutral",
        default_voice: str = "informative",
    ) -> PolicyRecommendation:
        stats = self._load_stats(intent)
        tone = stats.preferred_tone or default_tone
        voice = stats.preferred_voice or default_voice
        confidence = min(1.0, stats.sample_count / 10.0)
        notes = f"Based on {stats.sample_count} samples (avg reward {stats.average_reward:.2f})."
        return PolicyRecommendation(
            intent=intent,
            tone=tone,
            voice=voice,
            confidence=confidence,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _feedback_key(self, intent: str) -> str:
        return f"learning:feedback:{intent}"

    def _stats_key(self, intent: str) -> str:
        return f"learning:stats:{intent}"

    def _persist_feedback(self, record: FeedbackRecord) -> None:
        key = self._feedback_key(record.intent)
        existing = self.memory.get_memory(MemoryScope.LONG, key)
        history = existing.value if existing else []
        history.append(asdict(record))
        history = history[-self.max_history :]
        self.memory.set_memory(
            MemoryScope.LONG,
            key,
            history,
            category="learning",
            tags=[record.intent, record.sentiment],
            metadata={"samples": len(history)},
        )

    def _load_stats(self, intent: str) -> PolicyStats:
        key = self._stats_key(intent)
        record = self.memory.get_memory(MemoryScope.LONG, key)
        if record and isinstance(record.value, dict):
            data = record.value
            return PolicyStats(
                intent=intent,
                sample_count=data.get("sample_count", 0),
                cumulative_reward=data.get("cumulative_reward", 0.0),
                average_reward=data.get("average_reward", 0.0),
                preferred_tone=data.get("preferred_tone"),
                preferred_voice=data.get("preferred_voice"),
            )
        return PolicyStats(intent=intent)

    def _persist_stats(self, stats: PolicyStats) -> None:
        key = self._stats_key(stats.intent)
        metadata = {"updated_at": datetime.utcnow().isoformat()}
        self.memory.set_memory(
            MemoryScope.LONG,
            key,
            asdict(stats),
            category="learning",
            tags=[stats.intent],
            metadata=metadata,
        )


__all__ = ["AdaptiveLearner"]


