"""
Memory system module for storing and retrieving conversations and knowledge.

This module provides access to the memory system which includes:
- Working memory (short-term conversation context)
- Episodic memory (summarized past conversations)
- Semantic memory (knowledge base: docs, code, persona, facts)
- Intelligent retrieval and context building
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from memory_system.types import (
    ContextPack,
    MemoryChunk,
    MemoryChunkType,
    TaskState,
    Turn,
)
from memory_system.working_memory import WorkingMemory
from memory_system.log_store import JsonlLogStore, LogStore
from memory_system.vector_store import InMemoryVectorStore, VectorStore
from memory_system.episodic_store import EpisodicMemoryStore
from memory_system.semantic_store import SemanticMemoryStore
from memory_system.summarizer import EmbeddingClient, LLMClient, Summarizer
from memory_system.orchestrator import MemoryOrchestrator
from memory_system.manager import MemoryManager


def create_memory_manager(
    llm_client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    storage_dir: Path | str = Path("memory_data"),
    working_memory_max_turns: int = 20,
    token_budget: int = 4000,
    token_estimator: Optional[Callable[[str], int]] = None,
    system_instructions: Optional[list[str]] = None,
    persona_chunks_ids: Optional[list[str]] = None,
    use_orchestrator: bool = True,
) -> MemoryManager:
    """
    Create and configure a MemoryManager with default settings.

    Args:
        llm_client: Optional LLM client for summarization and orchestration
        embedding_client: Required embedding client for generating embeddings
        storage_dir: Directory for storing logs and memory chunks
        working_memory_max_turns: Maximum turns in working memory
        token_budget: Maximum tokens for context building
        token_estimator: Optional function to estimate tokens (default: word_count * 1.5)
        system_instructions: Optional system instructions
        persona_chunks_ids: Optional list of persona chunk IDs
        use_orchestrator: Whether to use orchestrator (requires llm_client)

    Returns:
        Configured MemoryManager instance

    Raises:
        ValueError: If embedding_client is None
    """
    if embedding_client is None:
        raise ValueError("embedding_client is required")

    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    working = WorkingMemory(max_turns=working_memory_max_turns)
    logs = JsonlLogStore(path=storage_path / "logs.jsonl")

    episodic_vs = InMemoryVectorStore()
    semantic_vs = InMemoryVectorStore()

    episodic_store = EpisodicMemoryStore(episodic_vs, storage_path / "episodic.jsonl")
    semantic_store = SemanticMemoryStore(semantic_vs, storage_path / "semantic.jsonl")

    orchestrator = None
    if use_orchestrator and llm_client:
        orchestrator = MemoryOrchestrator(llm=llm_client)

    manager = MemoryManager(
        working_memory=working,
        log_store=logs,
        episodic_store=episodic_store,
        semantic_store=semantic_store,
        orchestrator=orchestrator,
        embedding_client=embedding_client,
        system_instructions=system_instructions or [],
        persona_chunks_ids=persona_chunks_ids or [],
        token_budget=token_budget,
        token_estimator=token_estimator,
    )

    return manager


def add_document_chunk(
    manager: MemoryManager,
    title: str,
    text: str,
    chunk_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Add a document chunk to semantic memory.

    Args:
        manager: MemoryManager instance
        title: Title of the document
        text: Document content
        chunk_id: Optional chunk ID (auto-generated if not provided)
        metadata: Optional metadata dictionary

    Returns:
        The chunk ID
    """
    from datetime import datetime
    import uuid

    if chunk_id is None:
        chunk_id = str(uuid.uuid4())

    chunk = MemoryChunk(
        id=chunk_id,
        type=MemoryChunkType.DOC,
        title=title,
        text=text,
        metadata=metadata or {},
        timestamp=datetime.utcnow(),
    )

    manager.add_semantic_chunk(chunk)
    return chunk_id


def add_persona_chunk(
    manager: MemoryManager,
    title: str,
    text: str,
    chunk_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Add a persona chunk to semantic memory.

    Args:
        manager: MemoryManager instance
        title: Title of the persona information
        text: Persona content
        chunk_id: Optional chunk ID (auto-generated if not provided)
        metadata: Optional metadata dictionary

    Returns:
        The chunk ID
    """
    from datetime import datetime
    import uuid

    if chunk_id is None:
        chunk_id = str(uuid.uuid4())

    chunk = MemoryChunk(
        id=chunk_id,
        type=MemoryChunkType.PERSONA,
        title=title,
        text=text,
        metadata=metadata or {},
        timestamp=datetime.utcnow(),
    )

    manager.add_semantic_chunk(chunk)
    return chunk_id


def process_turn(
    manager: MemoryManager,
    session_id: str,
    role: str,
    content: str,
) -> None:
    """
    Process a conversation turn and add it to memory.

    Args:
        manager: MemoryManager instance
        session_id: Session identifier
        role: Turn role ("user", "assistant", or "system")
        content: Turn content
    """
    from datetime import datetime

    turn = Turn(role=role, content=content, timestamp=datetime.utcnow())
    manager.on_turn(session_id, turn)


def get_context(
    manager: MemoryManager,
    session_id: str,
    user_message: str,
) -> ContextPack:
    """
    Build context pack for a user message.

    Args:
        manager: MemoryManager instance
        session_id: Session identifier
        user_message: Current user message

    Returns:
        ContextPack with all relevant information
    """
    return manager.build_context(session_id, user_message)

