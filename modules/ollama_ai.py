"""
Ollama AI integration module.

This module provides Ollama-based LLM and embedding clients that can be used
with the memory system and other AI components.

Install dependencies:
    pip install ollama

The module uses the Ollama API to interact with local or remote Ollama instances.
Default model: P2Wdisabled/lyra:7b
"""

from __future__ import annotations

import os
from typing import Optional

try:  # pragma: no cover - optional dependency
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ollama = None  # type: ignore

from memory_system.summarizer import EmbeddingClient, LLMClient


class BackendUnavailable(RuntimeError):
    """Raised when Ollama is not available."""


class OllamaLLMClient(LLMClient):
    """
    LLM client implementation using Ollama.

    This client implements the LLMClient protocol and can be used with
    the memory system's orchestrator and summarizer.

    Example usage::

        from modules.ollama_ai import OllamaLLMClient

        client = OllamaLLMClient(model="P2Wdisabled/lyra:7b")
        response = client.generate("Hello, how are you?")
    """

    def __init__(
        self,
        model: str = "P2Wdisabled/lyra:7b",
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize Ollama LLM client.

        Args:
            model: Model name to use (default: P2Wdisabled/lyra:7b)
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        if ollama is None:
            raise BackendUnavailable(
                "Ollama is not installed. Install it with: pip install ollama"
            )

        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout
        
        # Set base URL if provided (ollama uses OLLAMA_HOST env var or default localhost)
        if base_url:
            os.environ["OLLAMA_HOST"] = base_url

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate text from a prompt using Ollama.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate (approximate)

        Returns:
            Generated text

        Raises:
            BackendUnavailable: If Ollama is not available
            RuntimeError: If the API call fails
        """
        if ollama is None:
            raise BackendUnavailable("Ollama is not installed")

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                },
            )
            # Handle both dict and object response formats
            if isinstance(response, dict):
                return response.get("response", "")
            else:
                return getattr(response, "response", "")
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with Ollama: {e}") from e


class OllamaEmbeddingClient(EmbeddingClient):
    """
    Embedding client implementation using Ollama.

    This client implements the EmbeddingClient protocol and can be used with
    the memory system for generating embeddings.

    Note: The model must support embeddings. If the model doesn't support
    embeddings, this will raise an error.

    Example usage::

        from modules.ollama_ai import OllamaEmbeddingClient

        client = OllamaEmbeddingClient(model="P2Wdisabled/lyra:7b")
        embedding = client.embed_text("Hello, world!")
    """

    def __init__(
        self,
        model: str = "P2Wdisabled/lyra:7b",
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize Ollama embedding client.

        Args:
            model: Model name to use (default: P2Wdisabled/lyra:7b)
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        if ollama is None:
            raise BackendUnavailable(
                "Ollama is not installed. Install it with: pip install ollama"
            )

        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout
        
        # Set base URL if provided (ollama uses OLLAMA_HOST env var or default localhost)
        if base_url:
            os.environ["OLLAMA_HOST"] = base_url

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for text using Ollama.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as a list of floats

        Raises:
            BackendUnavailable: If Ollama is not available
            RuntimeError: If the API call fails or model doesn't support embeddings
        """
        if ollama is None:
            raise BackendUnavailable("Ollama is not installed")

        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            # Handle both dict and object response formats
            if isinstance(response, dict):
                embedding = response.get("embedding", [])
            else:
                embedding = getattr(response, "embedding", [])
            
            if not embedding:
                raise RuntimeError(
                    f"Model {self.model} does not support embeddings or returned empty embedding"
                )
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding with Ollama: {e}") from e


class OllamaAISession:
    """
    High-level session for interacting with Ollama AI.

    This class provides a convenient interface for using Ollama with the memory system
    and for general AI interactions.

    Example usage::

        from modules.ollama_ai import OllamaAISession

        session = OllamaAISession()
        response = session.chat("Hello, how are you?")
    """

    def __init__(
        self,
        model: str = "P2Wdisabled/lyra:7b",
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize Ollama AI session.

        Args:
            model: Model name to use (default: P2Wdisabled/lyra:7b)
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            system_prompt: Optional system prompt to set
        """
        self.llm_client = OllamaLLMClient(model=model, base_url=base_url)
        self.embedding_client = OllamaEmbeddingClient(model=model, base_url=base_url)
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.system_prompt = system_prompt
        self._conversation_history: list[dict[str, str]] = []
        
        # Set base URL if provided
        if base_url:
            os.environ["OLLAMA_HOST"] = base_url

        if system_prompt:
            self._conversation_history.append({"role": "system", "content": system_prompt})

    def chat(self, message: str) -> str:
        """
        Send a chat message and get a response.

        Args:
            message: User message

        Returns:
            Assistant response
        """
        if ollama is None:
            raise BackendUnavailable("Ollama is not installed")

        # Add user message to history
        self._conversation_history.append({"role": "user", "content": message})

        try:
            # Get full response using ollama.chat()
            response = ollama.chat(
                model=self.model,
                messages=self._conversation_history,
            )
            # Handle both dict and object response formats
            if isinstance(response, dict):
                assistant_message = response.get("message", {}).get("content", "")
            else:
                # Try object attribute access
                try:
                    assistant_message = response.message.content
                except AttributeError:
                    # Fallback to dict access if object doesn't have the attribute
                    assistant_message = response.get("message", {}).get("content", "")
            self._conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            raise RuntimeError(f"Failed to chat with Ollama: {e}") from e

    def chat_stream(self, message: str):
        """
        Send a chat message and stream the response.

        Args:
            message: User message

        Yields:
            Response chunks as they arrive
        """
        if ollama is None:
            raise BackendUnavailable("Ollama is not installed")

        # Add user message to history
        self._conversation_history.append({"role": "user", "content": message})

        try:
            response_text = ""
            # Stream response using ollama.chat() with stream=True
            stream = ollama.chat(
                model=self.model,
                messages=self._conversation_history,
                stream=True,
            )
            for chunk in stream:
                # Handle both dict and object response formats
                if isinstance(chunk, dict):
                    content = chunk.get("message", {}).get("content", "")
                else:
                    # Try object attribute access
                    try:
                        content = chunk.message.content
                    except AttributeError:
                        # Fallback to dict access
                        content = chunk.get("message", {}).get("content", "")
                response_text += content
                yield content

            # Add assistant response to history
            self._conversation_history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            raise RuntimeError(f"Failed to chat with Ollama: {e}") from e

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
        if self.system_prompt:
            self._conversation_history.append({"role": "system", "content": self.system_prompt})

    def get_history(self) -> list[dict[str, str]]:
        """Get conversation history."""
        return self._conversation_history.copy()


def create_ollama_clients(
    model: str = "P2Wdisabled/lyra:7b",
    base_url: Optional[str] = None,
) -> tuple[OllamaLLMClient, OllamaEmbeddingClient]:
    """
    Create both LLM and embedding clients for use with the memory system.

    Args:
        model: Model name to use (default: P2Wdisabled/lyra:7b)
        base_url: Base URL for Ollama API (default: http://localhost:11434)

    Returns:
        Tuple of (LLMClient, EmbeddingClient)

    Example usage::

        from modules.ollama_ai import create_ollama_clients
        from modules.memory import create_memory_manager

        llm, emb = create_ollama_clients()
        manager = create_memory_manager(llm_client=llm, embedding_client=emb)
    """
    llm = OllamaLLMClient(model=model, base_url=base_url)
    emb = OllamaEmbeddingClient(model=model, base_url=base_url)
    return llm, emb

