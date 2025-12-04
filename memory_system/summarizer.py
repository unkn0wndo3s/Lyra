"""
Summarizer and LLM/embedding interfaces.
"""

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from memory_system.types import MemoryChunk, Turn


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        ...


class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as a list of floats
        """
        ...


class Summarizer:
    """
    Summarizer for conversations and memory chunks.
    Uses an LLM client to generate summaries.
    """

    def __init__(self, llm: LLMClient):
        """
        Initialize summarizer.

        Args:
            llm: LLM client for generating summaries
        """
        self.llm = llm

    def summarize_conversation(self, turns: list["Turn"]) -> str:
        """
        Summarize a conversation, highlighting goals, decisions, and user preferences.

        Args:
            turns: List of conversation turns

        Returns:
            Summary string
        """
        if not turns:
            return "No conversation to summarize."

        # Build conversation text
        conversation_text = "\n".join(
            f"{turn.role}: {turn.content}" for turn in turns
        )

        prompt = f"""Summarize the following conversation. Focus on:
- Key goals and objectives discussed
- Important decisions made
- User preferences and requirements
- Any specific facts or information that should be remembered
- Context that would be useful for future conversations

Conversation:
{conversation_text}

Summary:"""

        return self.llm.generate(prompt, max_tokens=512)

    def summarize_chunks(self, chunks: list["MemoryChunk"], max_tokens: int = 512) -> str:
        """
        Compress multiple memory chunks into a single summary.

        Args:
            chunks: List of memory chunks to summarize
            max_tokens: Maximum tokens for the summary

        Returns:
            Summary string
        """
        if not chunks:
            return "No chunks to summarize."

        chunks_text = "\n\n".join(
            f"[{chunk.type.value}] {chunk.title}\n{chunk.text}" for chunk in chunks
        )

        prompt = f"""Summarize and consolidate the following memory chunks into a coherent summary:

{chunks_text}

Consolidated Summary:"""

        return self.llm.generate(prompt, max_tokens=max_tokens)

