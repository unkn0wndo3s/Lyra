# Contextual Memory Retrieval

Lyra needs to surface the most relevant memories for each interaction. The
`MemoryRetriever` component ranks both short-term (JSON) and long-term (SQLite)
entries using keyword overlap, tag matches, recency, and optional importance
metadata.

## Usage

```python
from lyra import MemoryRetriever

retriever = MemoryRetriever()
memories = retriever.retrieve("status on deployment", tags=["status_request"])
for memory in memories:
    print(memory.scope, memory.key, memory.score)
```

Results combine short-term entries (recent conversation snippets) with
long-term knowledge (historical context). Scores float between 0 and 1.

## Scoring Factors

| Factor | Description |
|--------|-------------|
| Keyword overlap | Token intersection between the query and memory text. |
| Tag bonus | Boost when requested tags overlap with stored tags. |
| Recency | Recent updates receive higher weight (based on `metadata.updated_at`). |
| Importance | Optional `metadata["importance"]` scales the score. |

## Updating Memories

`MemoryRetriever.update_memory(...)` wraps `MemoryManager.set_memory` and
automatically stamps `updated_at` timestamps so new information is immediately
eligible for retrieval.

```python
retriever.update_memory(
    scope=MemoryScope.LONG,
    key="deployment:summary",
    value={"status": "green", "owner": "ops"},
    metadata={"importance": 1.5},
    tags=["status_request", "ops"],
)
```

## Prioritization

- Use the `importance` metadata field to prioritize mission-critical facts.
- Tag memories (e.g., `["user_profile", "preferences"]`) so queries can target
  specific domains.
- Short-term contexts are automatically pruned by TTL, ensuring the retriever
  favors the freshest interactions.


