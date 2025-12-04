# Memory System

A comprehensive memory system for storing and retrieving conversations and knowledge using summarization, embeddings, and vector search.

## Overview

The memory system provides:

- **Short-term "working memory"**: Recent conversation turns kept in memory
- **Long-term "episodic" memory**: Summarized past conversations stored with embeddings
- **Long-term "semantic" memory**: Knowledge base (docs, code, persona, facts) with vector search
- **Intelligent retrieval**: Optional LLM-based orchestrator for planning what to retrieve
- **Context building**: Automatic assembly of relevant context within token budgets

## Features

- **No hard dependencies on LLM providers**: Abstract interfaces allow you to plug in any model
- **Type-safe**: Full type hints throughout
- **Simple storage**: JSONL files for logs and chunks, in-memory NumPy-based vector store
- **Extensible**: Easy to swap in FAISS, Chroma, or other vector stores by implementing the `VectorStore` interface

## Installation

The package requires Python 3.10+ and the following dependencies:

- `numpy` (for vector similarity search)

Install with:

```bash
pip install numpy
```

## Quick Start

### 1. Implement LLM and Embedding Clients

First, implement the `LLMClient` and `EmbeddingClient` protocols:

```python
from memory_system.summarizer import LLMClient, EmbeddingClient

class MyLLMClient:
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Your LLM implementation here
        # Return generated text
        pass

class MyEmbeddingClient:
    def embed_text(self, text: str) -> list[float]:
        # Your embedding implementation here
        # Return embedding vector
        pass
```

### 2. Set Up the Memory System

```python
from pathlib import Path
from datetime import datetime
from memory_system.types import Turn, TaskState
from memory_system.working_memory import WorkingMemory
from memory_system.log_store import JsonlLogStore
from memory_system.vector_store import InMemoryVectorStore
from memory_system.episodic_store import EpisodicMemoryStore
from memory_system.semantic_store import SemanticMemoryStore
from memory_system.orchestrator import MemoryOrchestrator
from memory_system.manager import MemoryManager

# Initialize clients
llm = MyLLMClient()
emb = MyEmbeddingClient()

# Set up components
working = WorkingMemory(max_turns=20)
logs = JsonlLogStore(path=Path("logs.jsonl"))

episodic_vs = InMemoryVectorStore()
semantic_vs = InMemoryVectorStore()

episodic_store = EpisodicMemoryStore(episodic_vs, Path("episodic.jsonl"))
semantic_store = SemanticMemoryStore(semantic_vs, Path("semantic.jsonl"))

orchestrator = MemoryOrchestrator(llm=llm)  # Optional: can be None for heuristic-based retrieval

# Create memory manager
manager = MemoryManager(
    working_memory=working,
    log_store=logs,
    episodic_store=episodic_store,
    semantic_store=semantic_store,
    orchestrator=orchestrator,
    embedding_client=emb,
    system_instructions=["You are a helpful but blunt assistant."],
    token_budget=4000,
)
```

### 3. Use the Memory System

```python
session_id = "example-session"

# New user message
turn = Turn(
    role="user",
    content="Remind me what we decided about the 150M model training.",
    timestamp=datetime.utcnow()
)

# Process the turn
manager.on_turn(session_id, turn)

# Build context for the main model
context_pack = manager.build_context(session_id, turn.content)

# Use context_pack to build your prompt
# context_pack contains:
# - system_instructions
# - persona_snippets
# - task_state
# - episodic_snippets (relevant past conversations)
# - semantic_snippets (relevant knowledge)
# - recent_turns
```

### 4. Add Knowledge to Semantic Memory

```python
from memory_system.types import MemoryChunk, MemoryChunkType

# Add a document
doc_chunk = MemoryChunk(
    id="doc-001",
    type=MemoryChunkType.DOC,
    title="API Documentation",
    text="The API endpoint /v1/chat accepts...",
    timestamp=datetime.utcnow(),
)

manager.add_semantic_chunk(doc_chunk)

# Add persona information
persona_chunk = MemoryChunk(
    id="persona-001",
    type=MemoryChunkType.PERSONA,
    title="User Preferences",
    text="User prefers concise responses and technical details.",
    timestamp=datetime.utcnow(),
)

manager.add_semantic_chunk(persona_chunk)

# Reference persona chunks in manager
manager.persona_chunks_ids = ["persona-001"]
```

### 5. Periodic Summarization

Periodically summarize conversations to add to episodic memory:

```python
# After accumulating some turns, summarize
manager.periodic_summarize(session_id, chunk_size=20)
```

## Architecture

### Components

- **WorkingMemory**: Maintains a sliding window of recent turns
- **LogStore**: Append-only storage of all conversation turns
- **VectorStore**: Abstract interface for embedding storage and similarity search
- **EpisodicMemoryStore**: Stores summarized past conversations
- **SemanticMemoryStore**: Stores knowledge base (docs, code, persona, facts)
- **Summarizer**: Uses LLM to summarize conversations and chunks
- **MemoryOrchestrator**: Plans what to retrieve (LLM-based or heuristic)
- **MemoryManager**: Main API that coordinates all components

### Data Flow

1. **Input**: New conversation turn arrives
2. **Logging**: Turn is appended to log store
3. **Working Memory**: Turn is added to working memory (if user/assistant)
4. **Retrieval Planning**: Orchestrator determines what to retrieve
5. **Search**: Queries are embedded and searched in episodic/semantic stores
6. **Scoring**: Results are scored by similarity, recency, and type
7. **Context Building**: Relevant snippets are assembled within token budget
8. **Output**: ContextPack is returned for the main model

## Customization

### Custom Vector Store

Implement the `VectorStore` interface to use FAISS, Chroma, etc.:

```python
from memory_system.vector_store import VectorStore

class MyVectorStore(VectorStore):
    def add(self, id: str, embedding: list[float], metadata: dict | None = None) -> None:
        # Your implementation
        pass
    
    def search(self, query_embedding: list[float], top_k: int = 5, filters: dict | None = None) -> list[tuple[str, float]]:
        # Your implementation
        pass
    
    def get(self, id: str) -> dict | None:
        # Your implementation
        pass
```

### Custom Token Estimator

Provide a custom token estimator for better budget management:

```python
def my_token_estimator(text: str) -> int:
    # Use tiktoken, transformers tokenizer, etc.
    return len(text.split()) * 1.3  # Or use actual tokenizer

manager = MemoryManager(
    # ... other args ...
    token_estimator=my_token_estimator,
)
```

## Type System

All core types are defined in `memory_system.types`:

- `Turn`: Single conversation turn
- `TaskState`: Current task state and progress
- `MemoryChunk`: A chunk of memory with optional embedding
- `MemoryChunkType`: Enum for chunk types (EPISODIC, SEMANTIC, PERSONA, FACT, CODE, DOC)
- `SearchQuery`: Query for searching stores
- `SearchResult`: Result from a search
- `ContextPack`: Final assembled context
- `RetrievalPlan`: Plan for what to retrieve

## License

This package is provided as-is for use in your projects.

