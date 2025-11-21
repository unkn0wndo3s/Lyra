"""Dialogue management utilities built on top of the NLP processor."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from lyra.memory.manager import MemoryManager, MemoryScope

from .models import (
    DialogueContext,
    DialogueResponse,
    DialogueTurn,
    NLPResult,
)
from .processor import NLPProcessor


class DialogueManager:
    """Maintains conversational context and syncs it with Lyra's memory."""

    def __init__(
        self,
        *,
        nlp_processor: Optional[NLPProcessor] = None,
        memory_manager: Optional[MemoryManager] = None,
        context_window: int = 20,
        short_term_ttl: int = 900,
        long_term_retention: bool = True,
    ) -> None:
        self.processor = nlp_processor or NLPProcessor()
        self.memory_manager = memory_manager or MemoryManager()
        self.context_window = context_window
        self.short_term_ttl = short_term_ttl
        self.long_term_retention = long_term_retention

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_input(
        self,
        text: str,
        *,
        speaker: str = "user",
        session_id: str = "default",
    ) -> DialogueResponse:
        result = self.processor.analyze(text)
        context = self._load_context(session_id)

        turn = DialogueTurn(
            speaker=speaker,
            text=text,
            intent=result.intent.label,
            sentiment=result.sentiment.label,
            metadata={
                "confidence": result.intent.confidence,
                "polarity": result.sentiment.polarity,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        context.append(turn, max_turns=self.context_window)
        context.metadata["last_updated"] = datetime.utcnow().isoformat()
        if result.sentiment.tone:
            tone = result.sentiment.tone
            context.metadata["tone_hint"] = {
                "style": tone.style,
                "voice": tone.voice,
                "escalate": tone.escalate,
                "notes": tone.notes,
            }

        self._persist_context(session_id, context)
        if self.long_term_retention:
            self._log_long_term(session_id, turn, result)

        return DialogueResponse(session_id=session_id, nlp=result, context=context)

    def get_context(self, session_id: str = "default") -> DialogueContext:
        return self._load_context(session_id)

    def reset_context(self, session_id: str = "default") -> None:
        key = self._context_key(session_id)
        self.memory_manager.delete_memory(MemoryScope.SHORT, key)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _context_key(self, session_id: str) -> str:
        return f"dialogue:{session_id}"

    def _load_context(self, session_id: str) -> DialogueContext:
        key = self._context_key(session_id)
        record = self.memory_manager.get_memory(MemoryScope.SHORT, key)
        if record and isinstance(record.value, dict):
            return self._deserialize_context(record.value)
        return DialogueContext(session_id=session_id)

    def _persist_context(self, session_id: str, context: DialogueContext) -> None:
        payload = self._serialize_context(context)
        key = self._context_key(session_id)
        self.memory_manager.set_memory(
            MemoryScope.SHORT,
            key,
            payload,
            ttl_seconds=self.short_term_ttl,
        )

    def _serialize_context(self, context: DialogueContext) -> dict:
        return {
            "session_id": context.session_id,
            "turns": [
                {
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "intent": turn.intent,
                    "sentiment": turn.sentiment,
                    "metadata": turn.metadata,
                }
                for turn in context.turns
            ],
            "metadata": context.metadata,
        }

    def _deserialize_context(self, payload: dict) -> DialogueContext:
        context = DialogueContext(
            session_id=payload.get("session_id", "default"),
            metadata=payload.get("metadata", {}),
        )
        for turn_data in payload.get("turns", []):
            context.turns.append(
                DialogueTurn(
                    speaker=turn_data.get("speaker", "user"),
                    text=turn_data.get("text", ""),
                    intent=turn_data.get("intent", "unknown"),
                    sentiment=turn_data.get("sentiment", "neutral"),
                    metadata=turn_data.get("metadata", {}),
                )
            )
        return context

    def _log_long_term(
        self,
        session_id: str,
        turn: DialogueTurn,
        result: NLPResult,
    ) -> None:
        timestamp = datetime.utcnow().isoformat()
        key = f"dialogue:{session_id}:{timestamp}"
        payload = {
            "speaker": turn.speaker,
            "text": turn.text,
            "intent": turn.intent,
            "confidence": result.intent.confidence,
            "entities": [entity.__dict__ for entity in result.entities],
            "sentiment": turn.sentiment,
            "polarity": result.sentiment.polarity,
            "tone": (
                {
                    "style": result.sentiment.tone.style,
                    "voice": result.sentiment.tone.voice,
                    "notes": result.sentiment.tone.notes,
                    "escalate": result.sentiment.tone.escalate,
                }
                if result.sentiment.tone
                else None
            ),
            "metadata": turn.metadata,
        }
        self.memory_manager.set_memory(
            MemoryScope.LONG,
            key,
            payload,
            category="dialogue",
            tags=[turn.intent, turn.sentiment],
            metadata={
                "session_id": session_id,
                "timestamp": timestamp,
                "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            },
        )


__all__ = ["DialogueManager"]


