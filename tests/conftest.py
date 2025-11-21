import pytest

from lyra.memory.manager import MemoryManager


@pytest.fixture
def temp_memory_manager(tmp_path):
    short = tmp_path / "short_term.json"
    long = tmp_path / "long_term.db"
    manager = MemoryManager(short_term_path=short, long_term_path=long)
    yield manager
    manager.close()


@pytest.fixture
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


