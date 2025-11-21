import pytest

from lyra.memory.manager import MemoryManager


@pytest.fixture
def temp_memory_manager(tmp_path):
    short = tmp_path / "short_term.json"
    long = tmp_path / "long_term.db"
    manager = MemoryManager(short_term_path=short, long_term_path=long)
    yield manager
    manager.close()


