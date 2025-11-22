"""
Emotion interpretation helpers for speech-to-text (STT) pipelines.

The module provides lightweight heuristics that classify transcripts into
emotions and formats the result as phrases such as ``"this is **happy**
functionning"``. It is designed to be used alongside ``modules.live_speech`` so
that every partial or final transcript can immediately include emotional tone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from modules.live_speech import TranscriptEvent

DEFAULT_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "happy": ("happy", "joy", "glad", "great", "awesome", "excited", "love"),
    "sad": ("sad", "down", "unhappy", "depressed", "blue"),
    "angry": ("angry", "mad", "furious", "annoyed", "irritated", "upset"),
    "surprised": ("surprised", "shocked", "wow", "unbelievable"),
    "fearful": ("scared", "afraid", "fear", "terrified", "nervous"),
    "calm": ("calm", "relaxed", "easygoing", "chill"),
    "neutral": (),
}


def _format_phrase(emotion: str) -> str:
    """Return the canonical formatted string for a detected emotion."""
    return f"this is **{emotion}** functionning"


@dataclass(frozen=True)
class EmotionResult:
    """Container with the detected emotion and convenience metadata."""

    emotion: str
    confidence: float
    formatted: str
    source_text: str


class EmotionInterpreter:
    """Simple keyword-based emotion classifier for STT transcripts."""

    def __init__(
        self,
        keywords: Optional[Mapping[str, Iterable[str]]] = None,
        default_emotion: str = "neutral",
        min_confidence: float = 0.15,
    ) -> None:
        self._default_emotion = default_emotion
        self._min_confidence = min_confidence
        self._keywords: Dict[str, Tuple[str, ...]] = {
            emotion: tuple(token.lower() for token in tokens)
            for emotion, tokens in (keywords or DEFAULT_KEYWORDS).items()
        }

    def analyze(self, text: str) -> EmotionResult:
        """Assign an emotion to the given transcript text."""
        normalized = text.strip().lower()
        if not normalized:
            return EmotionResult(
                emotion=self._default_emotion,
                confidence=0.0,
                formatted=_format_phrase(self._default_emotion),
                source_text=text,
            )

        best_emotion = self._default_emotion
        best_score = 0

        for emotion, tokens in self._keywords.items():
            score = sum(1 for token in tokens if token and token in normalized)
            if score > best_score:
                best_score = score
                best_emotion = emotion

        confidence = max(
            self._min_confidence,
            min(1.0, best_score / max(1, len(self._keywords.get(best_emotion, (1,))))),
        )

        return EmotionResult(
            emotion=best_emotion,
            confidence=confidence,
            formatted=_format_phrase(best_emotion),
            source_text=text,
        )


def interpret_text(text: str, interpreter: Optional[EmotionInterpreter] = None) -> EmotionResult:
    """Analyze the provided text and return an EmotionResult."""
    interpreter = interpreter or EmotionInterpreter()
    return interpreter.analyze(text)


def emotion_aware_handler(
    *,
    on_transcript: Optional[Callable[[TranscriptEvent], None]] = None,
    on_emotion: Optional[Callable[[EmotionResult, TranscriptEvent], None]] = None,
    interpreter: Optional[EmotionInterpreter] = None,
) -> Callable[[TranscriptEvent], None]:
    """
    Build a callback compatible with ``LiveSpeechStreamer.start``.

    The returned handler forwards transcripts to ``on_transcript`` (if provided)
    and emits emotion results through ``on_emotion`` each time text is received.
    """

    interpreter = interpreter or EmotionInterpreter()

    def handler(event: TranscriptEvent) -> None:
        if on_transcript:
            on_transcript(event)
        if not event.text:
            return
        result = interpreter.analyze(event.text)
        if on_emotion:
            on_emotion(result, event)

    return handler

