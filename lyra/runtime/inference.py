"""Real-time inference service that ties the LLM into Lyra's memory + dialogue stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase

from lyra.memory.retrieval import MemoryRetriever
from lyra.nlp.dialogue import DialogueManager


@dataclass
class InferenceConfig:
    model_path: str
    tokenizer_path: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    device: Optional[str] = None
    context_turns: int = 6
    memory_results: int = 3


class RealTimeInferenceService:
    """Wraps a causal LM so it can converse with Lyra components."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        dialogue_manager: DialogueManager,
        memory_retriever: MemoryRetriever,
    ) -> None:
        self.config = config
        tokenizer_path = config.tokenizer_path or config.model_path
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None,
        )
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dialogue_manager = dialogue_manager
        self.memory_retriever = memory_retriever
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _build_context(self, session_id: str) -> str:
        context = self.dialogue_manager.get_context(session_id)
        turns = context.turns[-self.config.context_turns :]
        formatted_turns = []
        for turn in turns:
            speaker = "user" if turn.speaker == "user" else "assistant"
            formatted_turns.append(f"<{speaker}> {turn.text}")
        memories = self.memory_retriever.retrieve(
            " ".join(t.text for t in turns[-2:]),
            limit=self.config.memory_results,
        )
        memory_lines = [f"- {m.value}" for m in memories if isinstance(m.value, str)]
        memory_block = "\n".join(memory_lines)
        dialogue_block = "\n".join(formatted_turns)
        return (
            "<system>You are Lyra, a friendly AI focused on natural conversation.</system>\n"
            f"<memory>\n{memory_block}\n</memory>\n"
            f"{dialogue_block}\n<assistant>"
        )

    def generate_reply(
        self,
        *,
        session_id: str,
        user_text: str,
        speaker: str = "user",
    ) -> str:
        response = self.dialogue_manager.process_input(user_text, speaker=speaker, session_id=session_id)
        context_prompt = self._build_context(session_id)
        input_text = f"{context_prompt}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.dialogue_manager.context_window if hasattr(self.dialogue_manager, "context_window") else 1024,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                use_cache=True,
            )
        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        # Update dialogue with assistant reply
        self.dialogue_manager.process_input(generated, speaker="assistant", session_id=session_id)
        return generated.strip()


def load_inference_service(
    *,
    model_path: str,
    dialogue_manager: DialogueManager,
    memory_retriever: MemoryRetriever,
    tokenizer_path: Optional[str] = None,
) -> RealTimeInferenceService:
    config = InferenceConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    return RealTimeInferenceService(
        config=config,
        dialogue_manager=dialogue_manager,
        memory_retriever=memory_retriever,
    )


__all__ = ["RealTimeInferenceService", "InferenceConfig", "load_inference_service"]


