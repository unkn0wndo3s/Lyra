"""Real-time learning engine for updating memories on the fly."""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

from lyra.memory.manager import MemoryManager, MemoryScope
from lyra.memory.retrieval import MemoryRetriever
from lyra.nlp.models import DialogueResponse, NLPResult

from .models import LearnedFact


class RealTimeLearner:
    """Rule-based learner that extracts facts from incoming dialogue."""

    def __init__(
        self,
        *,
        memory_manager: Optional[MemoryManager] = None,
        retriever: Optional[MemoryRetriever] = None,
        default_ttl_seconds: int = 24 * 3600,
    ) -> None:
        self.memory = memory_manager or MemoryManager()
        self.retriever = retriever or MemoryRetriever(memory_manager=self.memory)
        self.default_ttl_seconds = default_ttl_seconds

    def learn_from_dialogue(
        self,
        response: DialogueResponse,
        *,
        session_id: str,
    ) -> List[LearnedFact]:
        facts = []
        facts.extend(self._extract_identity_rules(response.nlp))
        facts.extend(self._extract_status_rules(response, session_id))
        facts.extend(self._extract_entity_facts(response.nlp, session_id))
        for fact in facts:
            self._persist_fact(fact)
        return facts

    # ------------------------------------------------------------------ #
    # Rule extraction
    # ------------------------------------------------------------------ #

    def _extract_identity_rules(self, nlp: NLPResult) -> List[LearnedFact]:
        text_lower = nlp.text.lower()
        facts: List[LearnedFact] = []
        match = re.search(r"(?:my name is|call me)\s+([a-z]+)", text_lower)
        if match:
            name = match.group(1).strip().title()
            facts.append(
                LearnedFact(
                    key="user:profile:name",
                    value={"name": name},
                    category="user_profile",
                    tags=["user", "name"],
                    importance=2.0,
                )
            )
        return facts

    def _extract_status_rules(
        self,
        response: DialogueResponse,
        session_id: str,
    ) -> List[LearnedFact]:
        facts: List[LearnedFact] = []
        text_lower = response.nlp.text.lower()
        if "status" in text_lower and " is " in text_lower:
            segments = text_lower.split("status")
            if len(segments) > 1:
                descriptor = segments[-1].strip(" .!")
                facts.append(
                    LearnedFact(
                        key=f"session:{session_id}:status",
                        value={"status": descriptor, "source": "user"},
                        category="session_status",
                        tags=["status", session_id],
                        ttl_seconds=self.default_ttl_seconds,
                    )
                )
        return facts

    def _extract_entity_facts(
        self,
        nlp: NLPResult,
        session_id: str,
    ) -> List[LearnedFact]:
        facts: List[LearnedFact] = []
        for entity in nlp.entities:
            if entity.label in {"PERSON", "ORG", "GPE"}:
                key = f"session:{session_id}:entity:{entity.text.lower()}"
                facts.append(
                    LearnedFact(
                        key=key,
                        value={"entity": entity.text, "label": entity.label, "context": nlp.text},
                        category="named_entity",
                        tags=[entity.label.lower(), session_id],
                        ttl_seconds=self.default_ttl_seconds,
                    )
                )
        return facts

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def _persist_fact(self, fact: LearnedFact) -> None:
        metadata = fact.metadata.copy()
        metadata.setdefault("importance", fact.importance)
        metadata["updated_at"] = datetime.utcnow().isoformat()
        if fact.ttl_seconds:
            self.memory.set_memory(
                MemoryScope.SHORT,
                fact.key,
                fact.value,
                ttl_seconds=fact.ttl_seconds,
                metadata=metadata,
            )
        self.memory.set_memory(
            MemoryScope.LONG,
            fact.key,
            fact.value,
            category=fact.category,
            tags=fact.tags,
            metadata=metadata,
        )


__all__ = ["RealTimeLearner"]


