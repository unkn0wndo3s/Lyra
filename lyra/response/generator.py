"""Response generation orchestrator."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from lyra.nlp import DialogueManager, NLPProcessor

from .models import ResponseCandidate, ResponsePlan, ResponseRequest
from .strategies import BaseResponder, OllamaResponder, TemplateResponder

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses using configurable strategies plus dialogue context."""

    def __init__(
        self,
        *,
        dialogue_manager: Optional[DialogueManager] = None,
        nlp_processor: Optional[NLPProcessor] = None,
        responders: Optional[Iterable[BaseResponder]] = None,
    ) -> None:
        self.dialogue_manager = dialogue_manager or DialogueManager()
        self.nlp_processor = nlp_processor or NLPProcessor()
        self.responders: List[BaseResponder] = list(responders) if responders else [
            OllamaResponder(enabled=False),
            TemplateResponder(),
        ]

    def register_responder(self, responder: BaseResponder, *, priority: int = 0) -> None:
        self.responders.insert(priority, responder)

    def generate(
        self,
        text: str,
        *,
        session_id: str = "default",
        speaker: str = "user",
        persona: str = "assistant",
    ) -> ResponseCandidate:
        dialogue_response = self.dialogue_manager.process_input(
            text,
            speaker=speaker,
            session_id=session_id,
        )
        request = ResponseRequest(
            session_id=session_id,
            text=text,
            context=dialogue_response.context,
            nlp=dialogue_response.nlp,
            metadata={"speaker": speaker},
        )
        plan = ResponsePlan.from_request(
            request,
            sentiment=dialogue_response.nlp.sentiment,
            persona=persona,
        )
        candidate = self._run_strategies(plan)
        plan.selected_candidate = candidate
        return candidate

    def _run_strategies(self, plan: ResponsePlan) -> ResponseCandidate:
        errors = []
        for responder in self.responders:
            try:
                candidate = responder.generate(plan)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Responder %s failed", responder.name)
                errors.append(str(exc))
                continue
            if candidate:
                plan.alternatives.append(candidate)
                return candidate
        raise RuntimeError(f"No response could be generated. Errors: {errors}")


__all__ = ["ResponseGenerator"]


