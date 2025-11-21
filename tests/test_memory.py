from datetime import datetime, timedelta

from lyra.memory.manager import MemoryManager, MemoryScope
from lyra.memory.retrieval import MemoryRetriever


def test_memory_manager_roundtrip(temp_memory_manager: MemoryManager):
    manager = temp_memory_manager

    manager.set_memory(
        MemoryScope.SHORT,
        "conversation:last",
        {"text": "hello world"},
        ttl_seconds=60,
        metadata={"importance": 1.2},
    )
    short_record = manager.get_memory(MemoryScope.SHORT, "conversation:last")
    assert short_record is not None
    assert short_record.value["text"] == "hello world"

    manager.set_memory(
        MemoryScope.LONG,
        "fact:alpha",
        {"status": "green"},
        category="status",
        tags=["alpha"],
        metadata={"importance": 1.5, "updated_at": datetime.utcnow().isoformat()},
    )
    fact = manager.get_memory(MemoryScope.LONG, "fact:alpha")
    assert fact is not None
    assert fact.value["status"] == "green"


def test_memory_retriever_ranks_tagged_memories(temp_memory_manager: MemoryManager):
    manager = temp_memory_manager
    now = datetime.utcnow().isoformat()
    manager.set_memory(
        MemoryScope.LONG,
        "project:alpha",
        "Alpha deployment is healthy",
        category="project",
        tags=["project-alpha"],
        metadata={"importance": 2.0, "updated_at": now},
    )
    manager.set_memory(
        MemoryScope.LONG,
        "project:beta",
        "Beta deployment pending review",
        category="project",
        tags=["project-beta"],
        metadata={
            "importance": 1.0,
            "updated_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
        },
    )

    retriever = MemoryRetriever(memory_manager=manager)
    results = retriever.retrieve(
        "status update for alpha deployment", tags=["project-alpha"]
    )
    assert results
    assert results[0].key == "project:alpha"


