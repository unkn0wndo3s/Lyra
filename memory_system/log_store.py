"""
Raw log store for storing all conversations.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from memory_system.types import Turn


class LogStore(ABC):
    """Abstract interface for log storage."""

    @abstractmethod
    def append(self, session_id: str, turn: "Turn") -> None:
        """
        Append a turn to the log.

        Args:
            session_id: Session identifier
            turn: The turn to append
        """
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> list["Turn"]:
        """
        Get all turns for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of turns for the session
        """
        pass

    @abstractmethod
    def iter_all(self) -> Iterable[tuple[str, "Turn"]]:
        """
        Iterate over all turns across all sessions.

        Yields:
            Tuples of (session_id, turn)
        """
        pass


class JsonlLogStore(LogStore):
    """
    JSONL-based log store implementation.
    Stores all sessions in a single file with session_id included per entry.
    """

    def __init__(self, path: Path):
        """
        Initialize JSONL log store.

        Args:
            path: Path to the JSONL file
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize_turn(self, turn: "Turn") -> dict:
        """Serialize a Turn to a dictionary."""
        result = {
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp.isoformat(),
        }
        if turn.speaker is not None:
            result["speaker"] = turn.speaker
        return result

    def _deserialize_turn(self, data: dict) -> "Turn":
        """Deserialize a dictionary to a Turn."""
        return Turn(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            speaker=data.get("speaker"),  # Optional field for backward compatibility
        )

    def append(self, session_id: str, turn: "Turn") -> None:
        """Append a turn to the log."""
        try:
            entry = {
                "session_id": session_id,
                "turn": self._serialize_turn(turn),
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            raise RuntimeError(f"Failed to append to log store: {e}") from e

    def get_session(self, session_id: str) -> list["Turn"]:
        """Get all turns for a session."""
        if not self.path.exists():
            return []

        turns = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("session_id") == session_id:
                        turns.append(self._deserialize_turn(entry["turn"]))
        except Exception as e:
            raise RuntimeError(f"Failed to read from log store: {e}") from e

        return turns

    def iter_all(self) -> Iterable[tuple[str, "Turn"]]:
        """Iterate over all turns across all sessions."""
        if not self.path.exists():
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    session_id = entry.get("session_id", "")
                    turn = self._deserialize_turn(entry["turn"])
                    yield (session_id, turn)
        except Exception as e:
            raise RuntimeError(f"Failed to iterate log store: {e}") from e

