"""Core memory stores and manager for Lyra."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import LongTermMemoryRecord, ShortTermMemoryRecord


class MemoryScope(str, Enum):
    """Enumeration of supported memory scopes."""

    SHORT = "short"
    LONG = "long"


class ShortTermMemoryStore:
    """JSON-backed store for transient memories with optional TTL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_payload([])
        self.purge_expired()

    def _read_payload(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as stream:
            try:
                payload = json.load(stream)
            except json.JSONDecodeError:
                return []
        return payload.get("items", [])

    def _write_payload(self, items: Iterable[Dict[str, Any]]) -> None:
        payload = {"items": list(items)}
        with self.path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

    def purge_expired(self) -> int:
        """Remove expired short-term memories."""
        items = []
        removed = 0
        now = datetime.utcnow()
        for payload in self._read_payload():
            record = ShortTermMemoryRecord.from_payload(payload)
            if record.expires_at and record.expires_at < now:
                removed += 1
                continue
            items.append(record.to_payload())
        if removed:
            self._write_payload(items)
        return removed

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShortTermMemoryRecord:
        expires_at = (
            datetime.utcnow() + timedelta(seconds=ttl_seconds)
            if ttl_seconds
            else None
        )
        record = ShortTermMemoryRecord(
            key=key,
            value=value,
            metadata=metadata or {},
            expires_at=expires_at,
        )
        remaining = [
            ShortTermMemoryRecord.from_payload(payload)
            for payload in self._read_payload()
            if payload["key"] != key
        ]
        remaining.append(record)
        self._write_payload(r.to_payload() for r in remaining)
        return record

    def get(
        self, key: str, *, include_expired: bool = False
    ) -> Optional[ShortTermMemoryRecord]:
        for payload in self._read_payload():
            if payload["key"] != key:
                continue
            record = ShortTermMemoryRecord.from_payload(payload)
            if not include_expired and record.expires_at:
                if record.expires_at < datetime.utcnow():
                    return None
            return record
        return None

    def delete(self, key: str) -> bool:
        removed = False
        filtered = []
        for payload in self._read_payload():
            if payload["key"] == key:
                removed = True
                continue
            filtered.append(payload)
        if removed:
            self._write_payload(filtered)
        return removed

    def list(self) -> List[ShortTermMemoryRecord]:
        self.purge_expired()
        return [
            ShortTermMemoryRecord.from_payload(payload)
            for payload in self._read_payload()
        ]


class LongTermMemoryStore:
    """SQLite-backed store for persistent memories."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def set(
        self,
        key: str,
        value: Any,
        *,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LongTermMemoryRecord:
        record = self.get(key) or LongTermMemoryRecord(
            key=key,
            value=value,
            category=category,
            tags=tags or [],
            metadata=metadata or {},
        )
        record.value = value
        if category is not None:
            record.category = category
        if tags is not None:
            record.tags = tags
        if metadata is not None:
            record.metadata = metadata
        record.touch()
        payload = (
            record.key,
            json.dumps(record.value),
            record.category,
            json.dumps(record.tags),
            json.dumps(record.metadata),
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
        )
        self.connection.execute(
            """
            INSERT INTO memories (key, value, category, tags, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                category=excluded.category,
                tags=excluded.tags,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.connection.commit()
        return record

    def get(self, key: str) -> Optional[LongTermMemoryRecord]:
        cursor = self.connection.execute(
            "SELECT * FROM memories WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return LongTermMemoryRecord(
            key=row["key"],
            value=json.loads(row["value"]),
            category=row["category"],
            tags=json.loads(row["tags"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def delete(self, key: str) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM memories WHERE key = ?", (key,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def query(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[LongTermMemoryRecord]:
        clauses: List[str] = []
        params: List[Any] = []

        if category:
            clauses.append("category = ?")
            params.append(category)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = f"LIMIT {int(limit)}" if limit else ""
        cursor = self.connection.execute(
            f"SELECT * FROM memories {where_sql} ORDER BY updated_at DESC {limit_sql}",
            params,
        )
        rows = cursor.fetchall()
        return [
            LongTermMemoryRecord(
                key=row["key"],
                value=json.loads(row["value"]),
                category=row["category"],
                tags=json.loads(row["tags"] or "[]"),
                metadata=json.loads(row["metadata"] or "{}"),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()

    def collect_garbage(self, *, max_age_days: Optional[int] = None) -> int:
        """Delete long-term memories that are expired or stale.

        Expiry is determined via ``metadata.expires_at`` or an optional max age.
        """
        now = datetime.utcnow()
        stale_keys = set()

        if max_age_days is not None:
            cutoff = now - timedelta(days=max_age_days)
            cursor = self.connection.execute(
                "SELECT key FROM memories WHERE updated_at < ?", (cutoff.isoformat(),)
            )
            stale_keys.update(row["key"] for row in cursor.fetchall())

        cursor = self.connection.execute(
            "SELECT key, metadata FROM memories WHERE metadata IS NOT NULL"
        )
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"] or "{}")
            expires_at = metadata.get("expires_at")
            if not expires_at:
                continue
            try:
                expires_dt = datetime.fromisoformat(expires_at)
            except ValueError:
                continue
            if expires_dt < now:
                stale_keys.add(row["key"])

        if not stale_keys:
            return 0

        self.connection.executemany(
            "DELETE FROM memories WHERE key = ?", [(key,) for key in stale_keys]
        )
        self.connection.commit()
        return len(stale_keys)


class MemoryManager:
    """Facade that exposes a unified API for memory interactions."""

    def __init__(
        self,
        *,
        short_term_path: Optional[Path] = None,
        long_term_path: Optional[Path] = None,
    ) -> None:
        root = Path(short_term_path or Path("data/short_term_memory.json"))
        short_path = Path(short_term_path) if short_term_path else root
        long_path = (
            Path(long_term_path)
            if long_term_path
            else Path("data/long_term_memory.db")
        )
        self.short_store = ShortTermMemoryStore(short_path)
        self.long_store = LongTermMemoryStore(long_path)

    def set_memory(
        self,
        scope: MemoryScope,
        key: str,
        value: Any,
        **kwargs: Any,
    ):
        if scope is MemoryScope.SHORT:
            return self.short_store.set(key, value, **kwargs)
        if scope is MemoryScope.LONG:
            return self.long_store.set(key, value, **kwargs)
        raise ValueError(f"Unsupported scope: {scope}")

    def get_memory(
        self, scope: MemoryScope, key: str, **kwargs: Any
    ):
        if scope is MemoryScope.SHORT:
            return self.short_store.get(key, **kwargs)
        if scope is MemoryScope.LONG:
            return self.long_store.get(key)
        raise ValueError(f"Unsupported scope: {scope}")

    def delete_memory(self, scope: MemoryScope, key: str) -> bool:
        if scope is MemoryScope.SHORT:
            return self.short_store.delete(key)
        if scope is MemoryScope.LONG:
            return self.long_store.delete(key)
        raise ValueError(f"Unsupported scope: {scope}")

    def query_long_term(
        self, *, category: Optional[str] = None, tag: Optional[str] = None, limit: Optional[int] = None
    ) -> List[LongTermMemoryRecord]:
        return self.long_store.query(category=category, tag=tag, limit=limit)

    def list_short_term(self) -> List[ShortTermMemoryRecord]:
        return self.short_store.list()

    def close(self) -> None:
        self.long_store.close()

    def collect_garbage(
        self, *, long_term_max_age_days: Optional[int] = None
    ) -> Dict[str, int]:
        """Purge expired short-term entries and stale long-term memories."""
        short_removed = self.short_store.purge_expired()
        long_removed = self.long_store.collect_garbage(
            max_age_days=long_term_max_age_days
        )
        return {
            "short_removed": short_removed,
            "long_removed": long_removed,
        }


