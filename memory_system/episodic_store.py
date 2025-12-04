"""
Episodic memory store for summarized past conversations.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from memory_system.types import SearchResult

if TYPE_CHECKING:
    from memory_system.types import MemoryChunk, SearchQuery
    from memory_system.vector_store import VectorStore


class EpisodicMemoryStore:
    """
    Store for episodic memories (summarized past conversations).
    Uses a VectorStore for similarity search and JSONL for persistence.
    """

    def __init__(self, vector_store: "VectorStore", storage_path: Path):
        """
        Initialize episodic memory store.

        Args:
            vector_store: Vector store for similarity search
            storage_path: Path to JSONL file for storing MemoryChunks
        """
        self.vector_store = vector_store
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks_cache: dict[str, "MemoryChunk"] = {}

    def _serialize_chunk(self, chunk: "MemoryChunk") -> dict:
        """Serialize a MemoryChunk to a dictionary."""
        return {
            "id": chunk.id,
            "type": chunk.type.value,
            "title": chunk.title,
            "text": chunk.text,
            "metadata": chunk.metadata,
            "timestamp": chunk.timestamp.isoformat(),
            "embedding": chunk.embedding,
        }

    def _deserialize_chunk(self, data: dict) -> "MemoryChunk":
        """Deserialize a dictionary to a MemoryChunk."""
        from memory_system.types import MemoryChunkType
        from datetime import datetime

        return MemoryChunk(
            id=data["id"],
            type=MemoryChunkType(data["type"]),
            title=data["title"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            embedding=data.get("embedding"),
        )

    def add_chunk(self, chunk: "MemoryChunk") -> None:
        """
        Add a memory chunk to the store.

        Args:
            chunk: The memory chunk to add

        Raises:
            ValueError: If chunk.embedding is None
        """
        if chunk.embedding is None:
            raise ValueError("MemoryChunk must have an embedding to be added to episodic store")

        # Save to storage
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._serialize_chunk(chunk)) + "\n")
        except Exception as e:
            raise RuntimeError(f"Failed to save chunk to episodic store: {e}") from e

        # Add to vector store
        self.vector_store.add(chunk.id, chunk.embedding, {"type": chunk.type.value})

        # Cache in memory
        self._chunks_cache[chunk.id] = chunk

    def search(self, query: "SearchQuery", embedding: list[float]) -> list["SearchResult"]:
        """
        Search for similar memory chunks.

        Args:
            query: Search query
            embedding: Query embedding vector

        Returns:
            List of search results sorted by score
        """
        # Search vector store
        filters = query.filters
        if filters is None:
            filters = {}
        filters["type"] = "episodic"  # Ensure we only get episodic chunks

        vector_results = self.vector_store.search(embedding, top_k=query.top_k, filters=filters)

        # Load corresponding chunks
        results = []
        for chunk_id, score in vector_results:
            chunk = self._get_chunk(chunk_id)
            if chunk:
                results.append(SearchResult(chunk=chunk, score=score))

        return results

    def _get_chunk(self, chunk_id: str) -> "MemoryChunk | None":
        """Get a chunk by ID, loading from cache or storage if needed."""
        if chunk_id in self._chunks_cache:
            return self._chunks_cache[chunk_id]

        # Load from storage
        if not self.storage_path.exists():
            return None

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data["id"] == chunk_id:
                        chunk = self._deserialize_chunk(data)
                        self._chunks_cache[chunk_id] = chunk
                        return chunk
        except Exception as e:
            raise RuntimeError(f"Failed to load chunk from episodic store: {e}") from e

        return None

    def load_all(self) -> list["MemoryChunk"]:
        """
        Load all chunks from storage (for debugging or re-indexing).

        Returns:
            List of all memory chunks
        """
        if not self.storage_path.exists():
            return []

        chunks = []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    chunk = self._deserialize_chunk(data)
                    chunks.append(chunk)
                    self._chunks_cache[chunk.id] = chunk
        except Exception as e:
            raise RuntimeError(f"Failed to load chunks from episodic store: {e}") from e

        return chunks

