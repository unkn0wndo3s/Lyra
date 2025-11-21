"""High-level memory interface for Lyra agents."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .manager import MemoryManager, MemoryScope
from .models import LongTermMemoryRecord, ShortTermMemoryRecord


class MemoryType(str, Enum):
    """Labels that distinguish memory domains."""

    USER = "user"
    ENVIRONMENT = "environment"
    OPERATIONAL = "operational"
    OTHER = "other"


class MemoryInterface:
    """Public interface for adding, retrieving, and pruning memories."""

    def __init__(self, manager: Optional[MemoryManager] = None) -> None:
        self.manager = manager or MemoryManager()

    def add_memory(
        self,
        *,
        scope: MemoryScope,
        key: str,
        value: Any,
        memory_type: Optional[MemoryType] = None,
        ttl_seconds: Optional[int] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ):
        """Insert or update a memory in the requested scope."""
        meta = dict(metadata or {})
        if memory_type:
            meta.setdefault("memory_type", memory_type.value)

        if scope is MemoryScope.SHORT:
            return self.manager.set_memory(
                MemoryScope.SHORT,
                key,
                value,
                ttl_seconds=ttl_seconds,
                metadata=meta,
            )

        if ttl_seconds and not expires_at:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        if expires_at:
            meta["expires_at"] = (
                expires_at.isoformat()
                if isinstance(expires_at, datetime)
                else str(expires_at)
            )

        effective_category = category or (
            memory_type.value if memory_type else None
        )
        return self.manager.set_memory(
            MemoryScope.LONG,
            key,
            value,
            category=effective_category,
            tags=tags,
            metadata=meta,
        )

    def retrieve_memory(
        self,
        *,
        scope: MemoryScope,
        key: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories filtered by scope, type, or metadata."""
        if key:
            record = self.manager.get_memory(
                scope,
                key,
                include_expired=include_expired if scope is MemoryScope.SHORT else False,
            )
            return (
                [self._to_dict(record)]
                if record and self._matches(record, memory_type, tag)
                else []
            )

        if scope is MemoryScope.SHORT:
            records = self.manager.list_short_term()
        else:
            effective_category = category or (
                memory_type.value if memory_type else None
            )
            records = self.manager.query_long_term(
                category=effective_category,
                tag=tag,
                limit=limit,
            )

        filtered = [
            self._to_dict(record)
            for record in records
            if self._matches(record, memory_type, tag)
        ]
        if limit:
            return filtered[:limit]
        return filtered

    def garbage_collect(
        self,
        *,
        long_term_max_age_days: Optional[int] = None,
    ) -> Dict[str, int]:
        """Remove expired or stale memories across both stores."""
        return self.manager.collect_garbage(
            long_term_max_age_days=long_term_max_age_days
        )

    def delete_memory(self, scope: MemoryScope, key: str) -> bool:
        """Remove a memory record from the selected scope."""
        return self.manager.delete_memory(scope, key)

    def close(self) -> None:
        self.manager.close()

    @staticmethod
    def _to_dict(record: Any) -> Dict[str, Any]:
        data = asdict(record)
        return data

    @staticmethod
    def _matches(
        record: Any,
        memory_type: Optional[MemoryType],
        tag: Optional[str],
    ) -> bool:
        if memory_type:
            stored_type = record.metadata.get("memory_type")  # type: ignore[attr-defined]
            if stored_type != memory_type.value:
                return False
        if tag and isinstance(record, LongTermMemoryRecord):
            return tag in record.tags
        return True


