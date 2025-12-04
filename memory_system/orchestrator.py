"""
Memory orchestrator for planning retrieval operations.
"""

import json
import re
from typing import TYPE_CHECKING

from memory_system.types import RetrievalPlan, SearchQuery

if TYPE_CHECKING:
    from memory_system.summarizer import LLMClient
    from memory_system.types import TaskState, Turn


class MemoryOrchestrator:
    """
    Orchestrator that plans what to retrieve from memory stores.
    Can use an LLM for intelligent planning or fall back to heuristics.
    """

    def __init__(self, llm: "LLMClient | None" = None):
        """
        Initialize memory orchestrator.

        Args:
            llm: Optional LLM client for intelligent planning. If None, uses heuristics.
        """
        self.llm = llm

    def plan(
        self,
        recent_turns: list["Turn"],
        task_state: "TaskState | None",
        user_message: str,
    ) -> "RetrievalPlan":
        """
        Plan what to retrieve from memory stores.

        Args:
            recent_turns: Recent conversation turns
            task_state: Current task state
            user_message: Current user message

        Returns:
            Retrieval plan
        """
        if self.llm is not None:
            return self._plan_with_llm(recent_turns, task_state, user_message)
        else:
            return self._plan_with_heuristics(user_message)

    def _plan_with_llm(
        self,
        recent_turns: list["Turn"],
        task_state: "TaskState | None",
        user_message: str,
    ) -> "RetrievalPlan":
        """Plan retrieval using LLM."""
        # Build context for LLM
        recent_context = "\n".join(
            f"{turn.role}: {turn.content}" for turn in recent_turns[-5:]
        )

        task_context = ""
        if task_state:
            task_context = f"Current goal: {task_state.goal or 'None'}\n"
            if task_state.current_step:
                task_context += f"Current step: {task_state.current_step}\n"

        prompt = f"""You are a memory retrieval planner. Based on the conversation context and user message, determine what to retrieve from memory.

Recent conversation:
{recent_context}

Task state:
{task_context}

Current user message: {user_message}

Output a JSON object with the following structure:
{{
    "need_episodic": <bool>,
    "need_semantic": <bool>,
    "episodic_queries": [<list of query strings>],
    "semantic_queries": [<list of query strings>],
    "max_results_per_store": <int>
}}

Guidelines:
- need_episodic: true if the user is asking about past conversations, decisions, or previous context
- need_semantic: true if the user needs knowledge, facts, documentation, or code
- episodic_queries: list of search query strings for episodic memory (e.g., ["previous discussion about X", "what we decided about Y"])
- semantic_queries: list of search query strings for semantic memory (e.g., ["API documentation", "code examples"])
- max_results_per_store: number of results to retrieve per store (typically 3-10)

Output only valid JSON, no other text:"""

        try:
            response = self.llm.generate(prompt, max_tokens=512)
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            plan_data = json.loads(response)

            return RetrievalPlan(
                need_episodic=plan_data.get("need_episodic", False),
                need_semantic=plan_data.get("need_semantic", True),
                episodic_queries=[
                    SearchQuery(text=q, top_k=plan_data.get("max_results_per_store", 5))
                    for q in plan_data.get("episodic_queries", [])
                ],
                semantic_queries=[
                    SearchQuery(text=q, top_k=plan_data.get("max_results_per_store", 5))
                    for q in plan_data.get("semantic_queries", [])
                ],
                max_results_per_store=plan_data.get("max_results_per_store", 5),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fall back to heuristics if parsing fails
            return self._plan_with_heuristics(user_message)

    def _plan_with_heuristics(self, user_message: str) -> "RetrievalPlan":
        """Plan retrieval using simple heuristics."""
        message_lower = user_message.lower()

        # Check for episodic memory triggers
        episodic_keywords = ["before", "previous", "last time", "earlier", "we discussed", "we decided", "remember"]
        need_episodic = any(keyword in message_lower for keyword in episodic_keywords)

        # Always need semantic (knowledge base)
        need_semantic = True

        # Use user message as query
        query = SearchQuery(text=user_message, top_k=5)

        return RetrievalPlan(
            need_episodic=need_episodic,
            need_semantic=need_semantic,
            episodic_queries=[query] if need_episodic else [],
            semantic_queries=[query],
            max_results_per_store=5,
        )

