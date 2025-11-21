from lyra import AdaptiveLearner
from lyra.memory.manager import MemoryManager, MemoryScope
from lyra.nlp.dialogue import DialogueManager
from lyra.response.generator import ResponseGenerator
from lyra.response.strategies import TemplateResponder


def test_response_generator_returns_text(temp_memory_manager: MemoryManager):
    dialogue = DialogueManager(memory_manager=temp_memory_manager, short_term_ttl=120)
    adaptive = AdaptiveLearner(memory_manager=temp_memory_manager)
    generator = ResponseGenerator(
        dialogue_manager=dialogue,
        responders=[TemplateResponder()],
        adaptive_learner=adaptive,
    )

    reply = generator.generate("hello there", session_id="resp")
    assert isinstance(reply.text, str)
    assert reply.text

    generator.record_feedback(
        session_id="resp",
        interaction_id="resp-1",
        user_input="hello there",
        reward=1.0,
        nlp_intent="greeting",
        sentiment_label="positive",
    )
    stats = temp_memory_manager.get_memory(MemoryScope.LONG, "learning:stats:greeting")
    assert stats is not None
    assert stats.value["sample_count"] == 1

