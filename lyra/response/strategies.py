"""Response generation strategies."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import requests

from .models import ResponseCandidate, ResponsePlan, ResponseRequest, StyleGuide

logger = logging.getLogger(__name__)


class BaseResponder(ABC):
    name = "base"

    @abstractmethod
    def generate(self, plan: ResponsePlan) -> Optional[ResponseCandidate]:
        """Attempt to produce a response candidate."""


class TemplateResponder(BaseResponder):
    """Rule-based responder for low-latency fallback."""

    name = "template"

    def generate(self, plan: ResponsePlan) -> Optional[ResponseCandidate]:
        nlp = plan.request.nlp
        tone = plan.guidance
        intent = nlp.intent.label

        if intent == "greeting":
            text = self._greeting_response(tone)
        elif intent == "status_request":
            text = self._status_response(tone, plan.request.context)
        elif intent == "shutdown":
            text = "Understood. I'll stop the current operation. Let me know if you need anything else."
        elif tone.escalation:
            text = self._escalation_response(tone)
        else:
            text = self._generic_response(tone)

        metadata = {
            "intent": intent,
            "tone": tone.tone,
            "voice": tone.voice,
            "notes": tone.notes,
        }
        return ResponseCandidate(text=text, strategy=self.name, metadata=metadata)

    def _greeting_response(self, tone: StyleGuide) -> str:
        return f"Hi there! I'm here to help. {self._tone_tail(tone)}"

    def _status_response(self, tone: StyleGuide, context) -> str:
        last_turn = context.turns[-1] if context.turns else None
        if last_turn and last_turn.metadata.get("summary"):
            summary = last_turn.metadata["summary"]
            return f"Here's the latest update: {summary}"
        return "I'm compiling the latest status and will share it with you shortly."

    def _escalation_response(self, tone: StyleGuide) -> str:
        return (
            "I'm sorry you're experiencing trouble. "
            "I'll escalate this right away and stay with you until it's resolved."
        )

    def _generic_response(self, tone: StyleGuide) -> str:
        return (
            "Thanks for the update. Let's keep moving. "
            + self._tone_tail(tone)
        )

    def _tone_tail(self, tone: StyleGuide) -> str:
        if tone.tone == "positive":
            return "It sounds like things are going well."
        if tone.tone == "negative":
            return "I understand this is frustrating; I'm on it."
        return "Feel free to share more details if anything changes."


class OllamaResponder(BaseResponder):
    """LLM-backed responder that calls a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "llama3",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 30.0,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.enabled = enabled

    def generate(self, plan: ResponsePlan) -> Optional[ResponseCandidate]:
        if not self.enabled:
            return None
        prompt = self._build_prompt(plan)
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(
                self.endpoint,
                data=json.dumps(payload),
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response") or data.get("output") or ""
            if not text:
                return None
            metadata = {"model": self.model}
            return ResponseCandidate(text=text.strip(), strategy=self.name, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama responder failed: %s", exc)
            return None

    def _build_prompt(self, plan: ResponsePlan) -> str:
        tone = plan.guidance
        context_lines = [
            f"{turn.speaker}: {turn.text}"
            for turn in plan.request.context.turns[-6:]
        ]
        context = "\n".join(context_lines)
        prompt = f"""
You are Lyra, an AI assistant with a {tone.voice} voice and {tone.tone} tone.
Conversation history:
{context}
User: {plan.request.text}
Notes: {tone.notes}
Respond as {tone.persona}. Keep it concise but helpful.
"""
        return prompt.strip()


__all__ = ["BaseResponder", "TemplateResponder", "OllamaResponder"]


