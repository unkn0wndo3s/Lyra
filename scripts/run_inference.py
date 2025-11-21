"""Simple CLI loop to chat with the real-time inference service."""

from __future__ import annotations

import argparse

from lyra.memory.manager import MemoryManager
from lyra.memory.retrieval import MemoryRetriever
from lyra.nlp.dialogue import DialogueManager
from lyra.runtime.inference import InferenceConfig, RealTimeInferenceService


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to HF causal LM")
    parser.add_argument("--tokenizer_path", help="Tokenizer path if different")
    parser.add_argument("--session_id", default="demo")
    parser.add_argument("--short_term_path", help="Optional short-term memory override")
    parser.add_argument("--long_term_path", help="Optional long-term memory override")
    return parser.parse_args()


def main():
    args = parse_args()
    memory = MemoryManager(
        short_term_path=args.short_term_path,
        long_term_path=args.long_term_path,
    )
    dialogue = DialogueManager(memory_manager=memory, short_term_ttl=300)
    retriever = MemoryRetriever(memory_manager=memory)
    service = RealTimeInferenceService(
        config=InferenceConfig(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
        ),
        dialogue_manager=dialogue,
        memory_retriever=retriever,
    )
    print("Chat session started. Type 'exit' to quit.")
    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        reply = service.generate_reply(session_id=args.session_id, user_text=user_text)
        print(f"lyra> {reply}")


if __name__ == "__main__":
    main()


