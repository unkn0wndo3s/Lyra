"""
Memory Manager - Main high-level API for the memory system.
"""

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from memory_system.types import ContextPack, MemoryChunkType

if TYPE_CHECKING:
    from memory_system.episodic_store import EpisodicMemoryStore
    from memory_system.log_store import LogStore
    from memory_system.orchestrator import MemoryOrchestrator
    from memory_system.semantic_store import SemanticMemoryStore
    from memory_system.summarizer import EmbeddingClient, Summarizer
    from memory_system.types import MemoryChunk, TaskState, Turn
    from memory_system.working_memory import WorkingMemory


class MemoryManager:
    """
    Main high-level API for managing memory.
    Coordinates working memory, log store, episodic/semantic stores, and retrieval.
    """

    def __init__(
        self,
        working_memory: "WorkingMemory",
        log_store: "LogStore",
        episodic_store: "EpisodicMemoryStore",
        semantic_store: "SemanticMemoryStore",
        orchestrator: "MemoryOrchestrator | None",
        embedding_client: "EmbeddingClient",
        system_instructions: list[str] | None = None,
        persona_chunks_ids: list[str] | None = None,
        token_budget: int = 4000,
        token_estimator: Callable[[str], int] | None = None,
    ):
        """
        Initialize memory manager.

        Args:
            working_memory: Working memory instance
            log_store: Log store for raw conversations
            episodic_store: Episodic memory store
            semantic_store: Semantic memory store
            orchestrator: Optional orchestrator for retrieval planning
            embedding_client: Client for generating embeddings
            system_instructions: Optional system instructions
            persona_chunks_ids: Optional list of persona chunk IDs in semantic store
            token_budget: Maximum tokens for context
            token_estimator: Function to estimate tokens in text (default: word_count * 1.5)
        """
        self.working_memory = working_memory
        self.log_store = log_store
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.orchestrator = orchestrator
        self.embedding_client = embedding_client
        self.system_instructions = system_instructions or []
        self.persona_chunks_ids = persona_chunks_ids or []
        self.token_budget = token_budget

        if token_estimator is None:
            # Default: approximate tokens as word_count * 1.5
            self.token_estimator = lambda text: int(len(text.split()) * 1.5)
        else:
            self.token_estimator = token_estimator

        # Create summarizer (requires LLM, but we'll create it lazily if needed)
        self._summarizer: "Summarizer | None" = None

    def _get_summarizer(self) -> "Summarizer | None":
        """Get or create summarizer if orchestrator has LLM."""
        if self._summarizer is None and self.orchestrator and self.orchestrator.llm:
            from memory_system.summarizer import Summarizer
            self._summarizer = Summarizer(self.orchestrator.llm)
        return self._summarizer

    def on_turn(self, session_id: str, turn: "Turn") -> None:
        """
        Process a new conversation turn.

        Args:
            session_id: Session identifier
            turn: The conversation turn
        """
        # Append to log store
        self.log_store.append(session_id, turn)

        # Add to working memory if user or assistant
        if turn.role in ("user", "assistant"):
            self.working_memory.add_turn(turn)

    def periodic_summarize(
        self,
        session_id: str,
        chunk_size: int = 20,
        summarizer: "Summarizer | None" = None,
    ) -> None:
        """
        Periodically summarize conversation chunks and add to episodic memory.

        Args:
            session_id: Session identifier
            chunk_size: Number of turns per chunk
            summarizer: Optional summarizer (uses orchestrator's LLM if not provided)
        """
        summarizer = summarizer or self._get_summarizer()
        if not summarizer:
            raise ValueError("Summarizer required for periodic_summarize. Provide LLM to orchestrator or pass summarizer.")

        # Load session turns
        all_turns = self.log_store.get_session(session_id)
        if len(all_turns) < chunk_size:
            return  # Not enough turns to summarize

        # Find unsummarized windows
        # Simple approach: summarize every chunk_size turns
        # In a real system, you'd track which windows have been summarized
        for i in range(0, len(all_turns), chunk_size):
            chunk_turns = all_turns[i : i + chunk_size]
            if not chunk_turns:
                continue

            # Summarize the chunk
            summary_text = summarizer.summarize_conversation(chunk_turns)

            # Create memory chunk
            chunk_id = str(uuid.uuid4())
            chunk = MemoryChunk(
                id=chunk_id,
                type=MemoryChunkType.EPISODIC,
                title=f"Conversation summary {i // chunk_size + 1}",
                text=summary_text,
                timestamp=chunk_turns[-1].timestamp,
                metadata={"session_id": session_id, "turn_range": (i, i + chunk_size)},
            )

            # Generate embedding
            chunk.embedding = self.embedding_client.embed_text(summary_text)

            # Add to episodic store
            self.episodic_store.add_chunk(chunk)

    def add_semantic_chunk(self, chunk: "MemoryChunk") -> None:
        """
        Add a semantic chunk (docs, code, persona, facts) to semantic memory.

        Args:
            chunk: The memory chunk to add
        """
        # Generate embedding if missing
        if chunk.embedding is None:
            chunk.embedding = self.embedding_client.embed_text(chunk.text)

        # Add to semantic store
        self.semantic_store.add_chunk(chunk)

    def build_context(self, session_id: str, user_message: str) -> "ContextPack":
        """
        Build context pack for the model.

        Args:
            session_id: Session identifier
            user_message: Current user message

        Returns:
            Context pack with all relevant information
        """
        from memory_system.utils import (
            compute_combined_score,
            compute_recency_factor,
            get_type_weight,
            trim_to_token_budget,
        )

        # Get recent turns and task state
        recent_turns = self.working_memory.get_recent_turns()
        task_state = self.working_memory.get_task_state()

        # Get retrieval plan
        if self.orchestrator:
            plan = self.orchestrator.plan(recent_turns, task_state, user_message)
        else:
            # Default plan: search both stores
            from memory_system.types import RetrievalPlan, SearchQuery
            plan = RetrievalPlan(
                need_episodic=True,
                need_semantic=True,
                episodic_queries=[SearchQuery(text=user_message, top_k=5)],
                semantic_queries=[SearchQuery(text=user_message, top_k=5)],
                max_results_per_store=5,
            )

        # Collect search results
        episodic_results: list["SearchResult"] = []
        semantic_results: list["SearchResult"] = []

        if plan.need_episodic:
            for query in plan.episodic_queries:
                query_embedding = self.embedding_client.embed_text(query.text)
                results = self.episodic_store.search(query, query_embedding)
                episodic_results.extend(results)

        if plan.need_semantic:
            for query in plan.semantic_queries:
                query_embedding = self.embedding_client.embed_text(query.text)
                results = self.semantic_store.search(query, query_embedding)
                semantic_results.extend(results)

        # Score and sort results
        now = datetime.utcnow()
        scored_episodic = []
        for result in episodic_results:
            recency = compute_recency_factor(result.chunk.timestamp, now)
            type_weight = get_type_weight(result.chunk.type)
            combined_score = compute_combined_score(result.score, recency, type_weight)
            scored_episodic.append((combined_score, result))

        scored_semantic = []
        for result in semantic_results:
            recency = compute_recency_factor(result.chunk.timestamp, now)
            type_weight = get_type_weight(result.chunk.type)
            combined_score = compute_combined_score(result.score, recency, type_weight)
            scored_semantic.append((combined_score, result))

        # Sort by score and take top items
        scored_episodic.sort(key=lambda x: x[0], reverse=True)
        scored_semantic.sort(key=lambda x: x[0], reverse=True)

        top_episodic = [result for _, result in scored_episodic[:plan.max_results_per_store]]
        top_semantic = [result for _, result in scored_semantic[:plan.max_results_per_store]]

        # Load persona snippets
        persona_snippets = []
        for persona_id in self.persona_chunks_ids:
            chunk_data = self.semantic_store.vector_store.get(persona_id)
            if chunk_data:
                # Load full chunk
                chunk = self.semantic_store._get_chunk(persona_id)
                if chunk and chunk.type == MemoryChunkType.PERSONA:
                    persona_snippets.append(chunk.text)

        # Build sections
        sections = {
            "persona": persona_snippets,
            "semantic": [result.chunk.text for result in top_semantic],
            "episodic": [result.chunk.text for result in top_episodic],
            "recent_turns": [f"{turn.role}: {turn.content}" for turn in recent_turns],
        }

        # Trim to token budget
        trimmed = trim_to_token_budget(sections, self.token_budget, self.token_estimator)

        # Build context pack
        context_pack = ContextPack(
            system_instructions=self.system_instructions,
            persona_snippets=trimmed.get("persona", []),
            task_state=task_state,
            episodic_snippets=trimmed.get("episodic", []),
            semantic_snippets=trimmed.get("semantic", []),
            recent_turns=recent_turns[-len(trimmed.get("recent_turns", [])):] if trimmed.get("recent_turns") else [],
        )

        return context_pack

