"""Data models shared across Lyra's input handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class InputType(str, Enum):
    """Supported modalities for inbound data."""

    TEXT = "text"
    SPEECH = "speech"
    IMAGE = "image"
    SENSOR = "sensor"


@dataclass
class InputResult:
    """Represents a captured input payload plus metadata."""

    type: InputType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputCaptureError(RuntimeError):
    """Raised when an input source cannot be read or decoded."""



