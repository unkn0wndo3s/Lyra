# Core Memory Model

Lyra separates memory into **short-term** and **long-term** stores so the agent can
react to live context without losing historically relevant information.

## Short-Term Memory

- **Storage:** JSON file at `data/short_term_memory.json`.
- **Purpose:** Conversation context, ephemeral user preferences, transient task
  state.
- **Structure:**
  ```json
  {
    "items": [
      {
        "key": "conversation_context",
        "value": {"messages": [...]},
        "metadata": {"source": "chat"},
        "expires_at": "2025-11-21T21:45:00"
      }
    ]
  }
  ```
- Entries can define a TTL (in seconds). Expired entries are automatically
  purged whenever the store is read or listed.

## Long-Term Memory

- **Storage:** SQLite database at `data/long_term_memory.db`.
- **Purpose:** Durable knowledge such as user history, processed documents, or
  learned insights.
- **Schema (`memories` table):**

  | Column      | Type   | Description                                  |
  |-------------|--------|----------------------------------------------|
  | `key`       | TEXT   | Primary identifier (unique).                 |
  | `value`     | TEXT   | JSON payload of the memory contents.         |
  | `category`  | TEXT   | Optional semantic grouping (e.g. `user`).    |
  | `tags`      | TEXT   | JSON array of searchable tags.               |
  | `metadata`  | TEXT   | JSON blob for arbitrary attributes.          |
  | `created_at`| TEXT   | ISO timestamp for creation.                  |
  | `updated_at`| TEXT   | ISO timestamp for last update.               |

## Memory API

The `MemoryManager` exposes a simple interface:

- `set_memory(scope, key, value, **kwargs)` — create or update a record.
- `get_memory(scope, key, **kwargs)` — fetch a record.
- `delete_memory(scope, key)` — remove a record.
- `list_short_term()` — view all active short-term entries.
- `query_long_term(category=None, tag=None, limit=None)` — filter durable
  memories.
- `collect_garbage(long_term_max_age_days=None)` — purge expired entries.

Use `MemoryScope.SHORT` for transient data and `MemoryScope.LONG` for durable
entries. All methods return strongly typed records defined in
`lyra/memory/models.py`.

## Memory Interface

`lyra.memory.interface.MemoryInterface` builds on top of the manager to provide
semantically rich helpers:

- `add_memory(scope, key, value, memory_type=MemoryType.USER, ...)`
  automatically tags metadata, supports TTL for both short- and long-term
  memories, and writes `expires_at` timestamps.
- `retrieve_memory(scope, key=None, memory_type=None, tag=None, limit=None)`
  returns matching memories as dictionaries (filtering by type or tag).
- `delete_memory(scope, key)` removes a record via the underlying manager.
- `garbage_collect(long_term_max_age_days=None)` triggers cleanup across both
  stores. Short-term records rely on TTL while long-term records respect
  `metadata.expires_at` and optional staleness windows.

Available `MemoryType` values: `USER`, `ENVIRONMENT`, `OPERATIONAL`, `OTHER`.


