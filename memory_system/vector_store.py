"""
Vector store for embeddings storage and similarity search.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class VectorStore(ABC):
    """Abstract interface for vector storage and search."""

    @abstractmethod
    def add(self, id: str, embedding: list[float], metadata: dict | None = None) -> None:
        """
        Add an embedding to the store.

        Args:
            id: Unique identifier for the embedding
            embedding: The embedding vector
            metadata: Optional metadata to store with the embedding
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters (exact match)

        Returns:
            List of (id, score) pairs, sorted by score descending
        """
        pass

    @abstractmethod
    def get(self, id: str) -> dict | None:
        """
        Get stored metadata for an ID.

        Args:
            id: The identifier

        Returns:
            Dictionary with metadata, or None if not found
        """
        pass


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store using NumPy for similarity search.
    Uses cosine similarity for search.
    """

    def __init__(self):
        """Initialize the in-memory vector store."""
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._metadata: dict[str, dict] = {}

    def add(self, id: str, embedding: list[float], metadata: dict | None = None) -> None:
        """Add an embedding to the store."""
        embedding_array = np.array(embedding, dtype=np.float32)

        if id in self._metadata:
            # Update existing
            idx = self._ids.index(id)
            if self._embeddings is not None:
                self._embeddings[idx] = embedding_array
            self._metadata[id] = metadata or {}
        else:
            # Add new
            self._ids.append(id)
            if self._embeddings is None:
                self._embeddings = embedding_array.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, embedding_array.reshape(1, -1)])
            self._metadata[id] = metadata or {}

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings using cosine similarity."""
        if self._embeddings is None or len(self._ids) == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)  # Normalize

        # Normalize stored embeddings
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        normalized_embeddings = self._embeddings / (norms + 1e-8)

        # Compute cosine similarity
        similarities = np.dot(normalized_embeddings, query_vec)

        # Apply filters if provided
        if filters:
            filtered_indices = []
            for idx, id_val in enumerate(self._ids):
                metadata = self._metadata.get(id_val, {})
                matches = all(metadata.get(k) == v for k, v in filters.items())
                if matches:
                    filtered_indices.append(idx)
            if not filtered_indices:
                return []
            similarities = similarities[filtered_indices]
            filtered_ids = [self._ids[i] for i in filtered_indices]
        else:
            filtered_ids = self._ids

        # Get top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(filtered_ids[i], float(similarities[i])) for i in top_indices]

        return results

    def get(self, id: str) -> dict | None:
        """Get stored metadata for an ID."""
        return self._metadata.get(id)

