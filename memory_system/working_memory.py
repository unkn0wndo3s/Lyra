"""
Working memory for short-term conversation context.
"""

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory_system.types import TaskState, Turn


class WorkingMemory:
    """
    Short-term memory that maintains a sliding window of recent conversation turns.
    This is NOT persistent - it's just the recent window.
    """

    def __init__(self, max_turns: int = 20, task_state: "TaskState | None" = None):
        """
        Initialize working memory.

        Args:
            max_turns: Maximum number of turns to keep in memory
            task_state: Initial task state
        """
        self.max_turns = max_turns
        self._turns: deque["Turn"] = deque(maxlen=max_turns)
        self._task_state = task_state

    def add_turn(self, turn: "Turn") -> None:
        """
        Add a turn to working memory.

        Args:
            turn: The conversation turn to add
        """
        self._turns.append(turn)

    def set_task_state(self, task_state: "TaskState") -> None:
        """
        Set the current task state.

        Args:
            task_state: The task state to set
        """
        self._task_state = task_state

    def get_task_state(self) -> "TaskState | None":
        """
        Get the current task state.

        Returns:
            The current task state, or None if not set
        """
        return self._task_state

    def get_recent_turns(self) -> list["Turn"]:
        """
        Get all recent turns in working memory.

        Returns:
            List of recent turns
        """
        return list(self._turns)

    def clear(self) -> None:
        """Clear all turns from working memory."""
        self._turns.clear()
        self._task_state = None

