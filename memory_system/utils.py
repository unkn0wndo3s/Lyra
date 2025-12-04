"""
Utility functions for scoring and context building.
"""

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory_system.types import MemoryChunk, MemoryChunkType


def compute_recency_factor(timestamp: datetime, now: datetime) -> float:
    """
    Compute recency factor for a timestamp.
    Recent items get closer to 1.0, older items get decayed.

    Args:
        timestamp: The timestamp to evaluate
        now: Current time

    Returns:
        Recency factor between 0 and 1
    """
    delta = now - timestamp
    days = delta.total_seconds() / (24 * 3600)

    # Exponential decay: half-life of 30 days
    # Formula: e^(-lambda * days) where lambda = ln(2) / 30
    lambda_decay = 0.693147 / 30.0  # ln(2) / 30
    factor = 1.0 / (1.0 + lambda_decay * days)

    return max(0.0, min(1.0, factor))


def get_type_weight(chunk_type: "MemoryChunkType") -> float:
    """
    Get weight factor for a memory chunk type.

    Args:
        chunk_type: The type of memory chunk

    Returns:
        Weight factor
    """
    weights = {
        "PERSONA": 1.2,
        "FACT": 1.0,
        "DOC": 1.0,
        "CODE": 1.0,
        "EPISODIC": 0.9,
        "SEMANTIC": 1.0,
    }
    return weights.get(chunk_type.name, 1.0)


def compute_combined_score(
    similarity: float,
    recency: float,
    type_weight: float,
    w_sim: float = 0.6,
    w_recency: float = 0.3,
    w_type: float = 0.1,
) -> float:
    """
    Compute combined score from similarity, recency, and type weight.

    Args:
        similarity: Similarity score (0-1)
        recency: Recency factor (0-1)
        type_weight: Type weight factor
        w_sim: Weight for similarity (default 0.6)
        w_recency: Weight for recency (default 0.3)
        w_type: Weight for type (default 0.1)

    Returns:
        Combined score
    """
    # Normalize type_weight to 0-1 range (assuming max is around 1.2)
    normalized_type = (type_weight - 0.9) / (1.2 - 0.9) if type_weight > 0.9 else 0.0
    normalized_type = max(0.0, min(1.0, normalized_type))

    score = w_sim * similarity + w_recency * recency + w_type * normalized_type
    return max(0.0, min(1.0, score))


def trim_to_token_budget(
    sections: dict[str, list[str]],
    token_budget: int,
    estimate_tokens: Callable[[str], int],
    priority_order: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Trim sections to fit within token budget, respecting priority order.

    Args:
        sections: Dictionary mapping section names to lists of text snippets
        token_budget: Maximum tokens allowed
        estimate_tokens: Function to estimate tokens in a string
        priority_order: Order of sections by priority (default: persona > task_state > semantic > episodic > recent_turns)

    Returns:
        Trimmed sections dictionary
    """
    if priority_order is None:
        priority_order = ["persona", "task_state", "semantic", "episodic", "recent_turns"]

    trimmed: dict[str, list[str]] = {}
    used_tokens = 0

    # Process sections in priority order
    for section_name in priority_order:
        if section_name not in sections:
            continue

        trimmed[section_name] = []
        for snippet in sections[section_name]:
            snippet_tokens = estimate_tokens(snippet)
            if used_tokens + snippet_tokens <= token_budget:
                trimmed[section_name].append(snippet)
                used_tokens += snippet_tokens
            else:
                # Try to fit partial snippet if there's any budget left
                remaining_budget = token_budget - used_tokens
                if remaining_budget > 10:  # Only if meaningful space remains
                    # Truncate snippet (rough approximation)
                    words = snippet.split()
                    approx_chars_per_token = 4
                    max_chars = remaining_budget * approx_chars_per_token
                    truncated = " ".join(words[:max_chars // 10])  # Rough word-based truncation
                    if truncated:
                        trimmed[section_name].append(truncated + "...")
                break

    # Add any remaining sections that weren't in priority order
    for section_name, snippets in sections.items():
        if section_name not in trimmed:
            trimmed[section_name] = []

    return trimmed

