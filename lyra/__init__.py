"""Lyra core package exposing high-level interfaces."""

from .adaptive import AdaptiveLearner, FeedbackRecord, PolicyRecommendation, PolicyStats
from .input import (
    InputCaptureError,
    InputManager,
    InputResult,
    InputType,
    SpeechStreamHandle,
)
from .memory.interface import MemoryInterface, MemoryType
from .memory.manager import MemoryManager, MemoryScope
from .nlp import DialogueManager, NLPProcessor, SentimentAnalyzer, ToneDirective
from .response import (
    BaseResponder,
    OllamaResponder,
    ResponseCandidate,
    ResponseGenerator,
    ResponsePlan,
    ResponseRequest,
    StyleGuide,
    TemplateResponder,
)
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
    "NLPProcessor",
    "DialogueManager",
    "SentimentAnalyzer",
    "ToneDirective",
    "AdaptiveLearner",
    "FeedbackRecord",
    "PolicyStats",
    "PolicyRecommendation",
    "ResponseGenerator",
    "ResponseRequest",
    "ResponsePlan",
    "ResponseCandidate",
    "StyleGuide",
    "BaseResponder",
    "TemplateResponder",
    "OllamaResponder",
    "RealTimeProcessor",
    "InputSourceConfig",
]


