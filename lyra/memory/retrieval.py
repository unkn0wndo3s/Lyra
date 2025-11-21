"""Contextual memory retrieval helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Sequence

from .manager import MemoryManager, MemoryScope


@dataclass
class RetrievedMemory:
    key: str
    scope: MemoryScope
    value: Any
    metadata: dict
    score: float


class MemoryRetriever:
    """Ranks short- and long-term memories by relevance to a query."""

    def __init__(self, *, memory_manager: Optional[MemoryManager] = None) -> None:
        self.memory = memory_manager or MemoryManager()

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        categories: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> List[RetrievedMemory]:
        short_results = self._score_short_term(query)
        long_results = self._score_long_term(query, categories=categories, tags=tags)
        combined = sorted(short_results + long_results, key=lambda item: item.score, reverse=True)
        return combined[:limit]

    def update_memory(
        self,
        *,
        scope: MemoryScope,
        key: str,
        value: Any,
        metadata: Optional[dict] = None,
        ttl_seconds: Optional[int] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        metadata = metadata or {}
        metadata.setdefault("updated_at", datetime.utcnow().isoformat())
        if scope is MemoryScope.SHORT:
            self.memory.set_memory(
                MemoryScope.SHORT,
                key,
                value,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )
        else:
            self.memory.set_memory(
                MemoryScope.LONG,
                key,
                value,
                category=category,
                tags=tags,
                metadata=metadata,
            )

    # ------------------------------------------------------------------ #
    # Scoring helpers
    # ------------------------------------------------------------------ #

    def _score_short_term(self, query: str) -> List[RetrievedMemory]:
        results: List[RetrievedMemory] = []
        query_tokens = self._tokenize(query)
        for record in self.memory.list_short_term():
            text = self._stringify(record.value)
            tokens = self._tokenize(text)
            overlap = self._keyword_overlap(query_tokens, tokens)
            importance = float(record.metadata.get("importance", 1.0))
            recency = self._recency_bonus(record.metadata)
            score = overlap * 0.7 + recency * 0.2 + importance * 0.1
            if score <= 0:
                continue
            results.append(
                RetrievedMemory(
                    key=record.key,
                    scope=MemoryScope.SHORT,
                    value=record.value,
                    metadata=record.metadata,
                    score=score,
                )
            )
        return results

    def _score_long_term(
        self,
        query: str,
        *,
        categories: Optional[Sequence[str]],
        tags: Optional[Sequence[str]],
    ) -> List[RetrievedMemory]:
        query_tokens = self._tokenize(query)
        candidates: List[RetrievedMemory] = []
        categories = categories or [None]
        search_tags = tags or [None]
        seen_keys = set()
        for category in categories:
            for tag in search_tags:
                rows = self.memory.query_long_term(category=category, tag=tag, limit=50)
                for row in rows:
                    if row.key in seen_keys:
                        continue
                    seen_keys.add(row.key)
                    text = self._stringify(row.value)
                    tokens = self._tokenize(text)
                    overlap = self._keyword_overlap(query_tokens, tokens)
                    tag_bonus = self._tag_bonus(row.tags, tags)
                    importance = float(row.metadata.get("importance", 1.0))
                    recency = self._recency_bonus(row.metadata)
                    score = overlap * 0.6 + tag_bonus * 0.2 + recency * 0.1 + importance * 0.1
                    if score <= 0:
                        continue
                    candidates.append(
                        RetrievedMemory(
                            key=row.key,
                            scope=MemoryScope.LONG,
                            value=row.value,
                            metadata=row.metadata,
                            score=score,
                        )
                    )
        return candidates

    # ------------------------------------------------------------------ #
    # Utility functions
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, (str, int, float)):
            return str(value)
        return str(value)

    @staticmethod
    def _keyword_overlap(query_tokens: List[str], text_tokens: List[str]) -> float:
        if not query_tokens or not text_tokens:
            return 0.0
        query_set = set(query_tokens)
        text_set = set(text_tokens)
        overlap = len(query_set & text_set)
        return overlap / math.sqrt(len(query_set) * len(text_set))

    @staticmethod
    def _recency_bonus(metadata: dict) -> float:
        timestamp = metadata.get("updated_at") or metadata.get("timestamp")
        if not timestamp:
            return 0.0
        try:
            time = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.0
        delta = datetime.utcnow() - time
        hours = max(delta.total_seconds() / 3600.0, 1.0)
        bonus = max(0.0, 1.0 / hours)
        return min(bonus, 1.0)

    @staticmethod
    def _tag_bonus(memory_tags: List[str], query_tags: Optional[Sequence[str]]) -> float:
        if not memory_tags or not query_tags:
            return 0.0
        memory_set = set(memory_tags)
        query_set = set(tag for tag in query_tags if tag)
        if not query_set:
            return 0.0
        return len(memory_set & query_set) / len(query_set)


__all__ = ["MemoryRetriever", "RetrievedMemory"]


