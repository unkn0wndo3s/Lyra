import asyncio

from lyra import (
    AdaptiveLearner,
    DialogueManager,
    InputManager,
    InputType,
    MemoryRetriever,
    RealTimeLearner,
    ResponseGenerator,
    TemplateResponder,
)
from lyra.memory.manager import MemoryManager, MemoryScope
from lyra.runtime.processor import RealTimeProcessor


def test_conversation_learning_pipeline(temp_memory_manager: MemoryManager):
    dialogue = DialogueManager(memory_manager=temp_memory_manager, short_term_ttl=120)
    learner = RealTimeLearner(memory_manager=temp_memory_manager)
    adaptive = AdaptiveLearner(memory_manager=temp_memory_manager)
    generator = ResponseGenerator(
        dialogue_manager=dialogue,
        responders=[TemplateResponder()],
        adaptive_learner=adaptive,
    )
    retriever = MemoryRetriever(memory_manager=temp_memory_manager)

    utterances = [
        "My name is Taylor",
        "The deployment status is green",
        "Thanks for the help!",
    ]

    for idx, text in enumerate(utterances):
        response = dialogue.process_input(text, session_id="integration")
        learner.learn_from_dialogue(response, session_id="integration")
        reply = generator.generate(text, session_id="integration")
        generator.record_feedback(
            session_id="integration",
            interaction_id=f"integration-{idx}",
            user_input=text,
            reward=1.0,
            nlp_intent=response.nlp.intent.label,
            sentiment_label=response.nlp.sentiment.label,
        )
        assert reply.text

    user_fact = retriever.retrieve("Taylor", tags=["user"])
    assert user_fact, "Expected learned user name to be retrievable"
    stats = temp_memory_manager.get_memory(
        MemoryScope.LONG, "learning:stats:statement"
    )
    assert stats is not None


def test_real_time_processor_handles_text_and_sensor():
    inputs = InputManager()
    loop = asyncio.new_event_loop()
    processor = RealTimeProcessor(input_manager=inputs, loop=loop)
    captured = []

    commands = iter(["task one", "task two"])
    processor.register_text_source(
        "scripted",
        supplier=lambda: next(commands),
        interval=0.01,
        max_events=2,
    )

    inputs.register_sensor(name="thermo", capture_fn=lambda: {"temp": 72})
    processor.register_sensor_stream(
        "thermo-stream",
        sensor_name="thermo",
        interval=0.01,
        max_events=2,
    )

    async def handle(result):
        captured.append(result.type)

    processor.register_handler(InputType.TEXT, handle)
    processor.register_handler(InputType.SENSOR, handle)

    try:
        loop.run_until_complete(processor.run_for(0.2))
    finally:
        loop.close()

    assert InputType.TEXT in captured
    assert InputType.SENSOR in captured

