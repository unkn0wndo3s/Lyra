"""
Core data models for the memory system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryChunkType(Enum):
    """Types of memory chunks."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERSONA = "persona"
    FACT = "fact"
    CODE = "code"
    DOC = "doc"


@dataclass
class Turn:
    """A single conversation turn."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    speaker: str | None = None  # Speaker identifier (from voice_identity)


@dataclass
class TaskState:
    """Current task state and progress."""
    goal: str | None = None
    subtasks: list[str] = field(default_factory=list)
    current_step: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryChunk:
    """A chunk of memory with optional embedding."""
    id: str
    type: MemoryChunkType
    title: str
    text: str  # summary or document chunk
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    embedding: list[float] | None = None


@dataclass
class SearchQuery:
    """Query for searching memory stores."""
    text: str
    top_k: int = 5
    filters: dict[str, Any] | None = None


@dataclass
class SearchResult:
    """Result from a memory store search."""
    chunk: MemoryChunk
    score: float


@dataclass
class ContextPack:
    """Final context assembled for the model."""
    system_instructions: list[str] = field(default_factory=list)
    persona_snippets: list[str] = field(default_factory=list)
    task_state: TaskState | None = None
    episodic_snippets: list[str] = field(default_factory=list)
    semantic_snippets: list[str] = field(default_factory=list)
    recent_turns: list[Turn] = field(default_factory=list)


@dataclass
class RetrievalPlan:
    """Plan for what to retrieve from memory stores."""
    need_episodic: bool = False
    need_semantic: bool = False
    episodic_queries: list[SearchQuery] = field(default_factory=list)
    semantic_queries: list[SearchQuery] = field(default_factory=list)
    max_results_per_store: int = 5

