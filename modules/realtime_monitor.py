"""
Realtime helper that fuses speech-to-text, emotion analysis, and speaker ID.

Use ``start_insight_session`` to launch a pipeline that listens to the
microphone, streams partial/final transcripts, tags each one with an inferred
emotion, and (optionally) figures out whether the speaker is already known via
``VoiceIdentifier``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional

try:  # pragma: no cover - optional dependency for speaker tracking
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

from modules.emotion_interpreter import EmotionInterpreter, EmotionResult
from modules.live_speech import LiveSpeechStreamer, TranscriptEvent
from modules.voice_identity import (
    IdentificationResult,
    VoiceIdentifier,
    BackendUnavailable as VoiceIdentityBackendUnavailable,
)

# Try to import Whisper streamer
try:
    from modules.whisper_live import WhisperLiveStreamer  # type: ignore
except ImportError:
    WhisperLiveStreamer = None  # type: ignore


@dataclass
class Insight:
    """Combined view of the live transcription pipeline."""

    text: str
    is_final: bool
    emotion: EmotionResult
    speaker: Optional[str]
    speaker_confidence: float
    is_known_speaker: bool
    speaker_scores: Dict[str, float] = field(default_factory=dict)


class _SpeakerTracker:
    """Maintains a rolling window of identification attempts."""

    def __init__(self, identifier: Optional[VoiceIdentifier], window: int = 6) -> None:
        self.identifier = identifier
        self.window: Deque[IdentificationResult] = deque(maxlen=window)
        self.latest: Optional[IdentificationResult] = None
        self.enabled = identifier is not None

    def submit(self, chunk: bytes, sample_rate: int) -> None:
        if not self.enabled or not self.identifier or np is None:
            return

        audio = np.frombuffer(chunk, dtype="<i2>").astype(np.float32)
        if not audio.size:
            return
        audio /= 32768.0

        try:
            result = self.identifier.identify(audio, sample_rate=sample_rate)
        except VoiceIdentityBackendUnavailable:
            self.enabled = False
            return

        self.window.append(result)
        self.latest = self._aggregate()

    def _aggregate(self) -> Optional[IdentificationResult]:
        if not self.window:
            return None
        return max(self.window, key=lambda r: r.confidence)


class LiveInsightSession:
    """Coordinates the live streaming pipeline."""

    def __init__(
        self,
        *,
        on_insight: Optional[Callable[[Insight], None]] = None,
        streamer: Optional[LiveSpeechStreamer] = None,
        emotion_interpreter: Optional[EmotionInterpreter] = None,
        voice_identifier: Optional[VoiceIdentifier] = None,
        smoothing_window: int = 6,
        stream_kwargs: Optional[dict] = None,
        use_whisper: bool = False,
        whisper_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Initialize the live insight session.

        Args:
            use_whisper: If True, use Whisper (large-v3) for better accuracy.
                Requires faster-whisper. Falls back to Vosk if unavailable.
            whisper_kwargs: Additional arguments for WhisperLiveStreamer
                (model_size, device, compute_type, etc.).
        """
        if use_whisper and WhisperLiveStreamer is not None:
            whisper_opts = whisper_kwargs or {}
            self.streamer = WhisperLiveStreamer(**whisper_opts)
        else:
            if use_whisper:
                print("Warning: Whisper not available, falling back to Vosk")
            self.streamer = streamer or LiveSpeechStreamer(**(stream_kwargs or {}))
        self.emotion_interpreter = emotion_interpreter or EmotionInterpreter()
        self.voice_identifier = voice_identifier
        self.on_insight = on_insight or self._default_sink
        self._speaker_tracker = _SpeakerTracker(voice_identifier, window=smoothing_window)
        self._active = False

    # ----------------------------------------------------------------- lifecycle
    def start(self, *, emit_partials: bool = True) -> "LiveInsightSession":
        if self._active:
            return self
        self.streamer.start(
            self._handle_transcript,
            emit_partials=emit_partials,
            on_audio_chunk=self._speaker_tracker.submit,
        )
        self._active = True
        return self

    def stop(self) -> None:
        if not self._active:
            return
        self.streamer.stop()
        self._active = False

    def __enter__(self) -> "LiveInsightSession":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ---------------------------------------------------------------- callbacks
    def _handle_transcript(self, event: TranscriptEvent) -> None:
        if not event.text:
            return

        emotion = self.emotion_interpreter.analyze(event.text)
        speaker = self._speaker_tracker.latest

        insight = Insight(
            text=event.text,
            is_final=event.is_final,
            emotion=emotion,
            speaker=speaker.speaker if speaker else None,
            speaker_confidence=speaker.confidence if speaker else 0.0,
            is_known_speaker=bool(speaker and speaker.is_known),
            speaker_scores=speaker.scores if speaker else {},
        )
        self.on_insight(insight)

    @staticmethod
    def _default_sink(insight: Insight) -> None:
        stamp = "FINAL" if insight.is_final else "LIVE"
        speaker = insight.speaker or "Unknown"
        print(f"[{stamp}] {speaker}: {insight.text} | {insight.emotion.formatted}")


def start_insight_session(**kwargs) -> LiveInsightSession:
    """
    Convenience helper to spin up ``LiveInsightSession`` with default settings.

    Returns the started session so the caller can later invoke ``stop()``.
    """
    session = LiveInsightSession(**kwargs)
    return session.start()

