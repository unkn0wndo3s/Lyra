"""Dataclasses that describe Lyra's memory records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


@dataclass
class ShortTermMemoryRecord:
    """Represents an ephemeral memory entry that can expire."""

    key: str
    value: Any
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly payload."""
        return {
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ShortTermMemoryRecord":
        expires_at = payload.get("expires_at")
        return cls(
            key=payload["key"],
            value=payload["value"],
            metadata=payload.get("metadata", {}),
            expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
        )


@dataclass
class LongTermMemoryRecord:
    """Represents a persistent memory entry stored in SQLite."""

    key: str
    value: Any
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp."""
        self.updated_at = _utcnow()


