"""Lyra core package exposing high-level interfaces."""

from .input import (
    InputCaptureError,
    InputManager,
    InputResult,
    InputType,
    SpeechStreamHandle,
)
from .memory.interface import MemoryInterface, MemoryType
from .memory.manager import MemoryManager, MemoryScope
from .runtime import InputSourceConfig, RealTimeProcessor

__all__ = [
    "InputManager",
    "InputType",
    "InputResult",
    "InputCaptureError",
    "SpeechStreamHandle",
    "MemoryManager",
    "MemoryScope",
    "MemoryInterface",
    "MemoryType",
    "RealTimeProcessor",
    "InputSourceConfig",
]


