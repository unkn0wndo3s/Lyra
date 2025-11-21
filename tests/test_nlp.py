from lyra.memory.manager import MemoryManager, MemoryScope
from lyra.nlp.dialogue import DialogueManager
from lyra.nlp.processor import NLPProcessor


def test_nlp_processor_produces_structured_output():
    processor = NLPProcessor()
    text = "Hello Lyra, can you share the status?"
    result = processor.analyze(text)

    assert result.text == text
    assert result.tokens, "Expected tokens to be populated"
    assert result.intent.label
    assert result.sentiment.label in {"positive", "negative", "neutral"}


def test_dialogue_manager_persists_context(temp_memory_manager: MemoryManager):
    dialogue = DialogueManager(memory_manager=temp_memory_manager, short_term_ttl=120)
    dialogue.process_input("Hello there", session_id="ctx")
    response = dialogue.process_input("Please give me the status", session_id="ctx")

    assert len(response.context.turns) == 2
    context_record = temp_memory_manager.get_memory(
        MemoryScope.SHORT, "dialogue:ctx"
    )
    assert context_record is not None
    assert len(context_record.value["turns"]) == 2

