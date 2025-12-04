"""
Live speech-to-text using OpenAI Whisper (large model) for maximum accuracy.

This module provides a high-accuracy alternative to Vosk, using Whisper's large
model for superior transcription quality. It processes audio chunks in near-real-time
for live transcription.

Install dependencies:
    pip install faster-whisper sounddevice torch

The module automatically downloads the Whisper large-v3 model on first use.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None  # type: ignore


@dataclass(frozen=True)
class TranscriptEvent:
    """Represents a chunk of recognized speech."""

    text: str
    is_final: bool


class BackendUnavailable(RuntimeError):
    """Raised when required dependencies are missing."""


class WhisperLiveStreamer:
    """
    High-accuracy live speech-to-text using OpenAI Whisper large model.

    This streamer processes audio chunks using Whisper's large-v3 model for
    superior accuracy compared to Vosk, with near-real-time transcription.

    Example usage::

        from modules.whisper_live import WhisperLiveStreamer

        def on_event(event):
            print("FINAL" if event.is_final else "LIVE", event.text)

        streamer = WhisperLiveStreamer()
        streamer.start(on_event)
        # ... later ...
        streamer.stop()
    """

    def __init__(
        self,
        *,
        model_size: str = "large-v3-turbo",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16_000,
        chunk_duration: float = 3.0,
        overlap: float = 0.5,
        language: Optional[str] = "en",
        beam_size: int = 5,
        silence_duration: float = 1.5,
        silence_threshold: float = 0.01,
    ) -> None:
        """
        Initialize the Whisper live streamer.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo").
                Defaults to "large-v3" for best accuracy. "large-v3-turbo" is faster with similar accuracy.
            device: "cpu" or "cuda" for GPU acceleration.
            compute_type: "int8", "int8_float16", "int16", "float16", or "float32".
                Lower precision = faster but less accurate.
            sample_rate: Audio sample rate in Hz.
            chunk_duration: Duration of audio chunks to process (seconds).
            overlap: Overlap between chunks (seconds) to avoid cutting words.
            language: Language code ("en", "fr", etc.) or None for auto-detection.
            beam_size: Beam search size (higher = more accurate but slower).
            silence_duration: Seconds of silence to wait before processing (default: 1.5).
            silence_threshold: Audio amplitude threshold for silence detection (default: 0.01).
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.language = language
        self.beam_size = beam_size
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold

        self._audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._model: Optional[WhisperModel] = None
        self._on_audio_chunk: Optional[Callable[[bytes, int], None]] = None

    def start(
        self,
        on_transcript: Callable[[TranscriptEvent], None],
        *,
        emit_partials: bool = True,
        on_audio_chunk: Optional[Callable[[bytes, int], None]] = None,
    ) -> None:
        """
        Begin streaming and invoke ``on_transcript`` as text is recognized.

        Args:
            on_transcript: Callback for transcript events.
            emit_partials: Whether to emit partial (non-final) transcripts.
            on_audio_chunk: Optional callback for raw audio chunks (bytes, sample_rate).
        """
        if self._thread and self._thread.is_alive():
            raise RuntimeError("WhisperLiveStreamer is already running.")

        self._prepare_backend()
        self._stop_event.clear()
        self._on_audio_chunk = on_audio_chunk
        self._thread = threading.Thread(
            target=self._run_stream,
            args=(on_transcript, emit_partials),
            daemon=True,
            name="WhisperLiveStreamer",
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop streaming and wait for the worker thread to finish."""
        if not self._thread:
            return
        self._stop_event.set()
        self._audio_queue.put_nowait(b"")  # unblock queue
        self._thread.join(timeout=timeout)
        self._thread = None
        self._stop_event.clear()

    def __enter__(self) -> "WhisperLiveStreamer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _prepare_backend(self) -> None:
        """Ensure Whisper and sounddevice are available."""
        if sd is None:
            raise BackendUnavailable(
                "sounddevice is required. Install it with `pip install sounddevice`."
            )
        if WhisperModel is None:
            raise BackendUnavailable(
                "faster-whisper is required. Install it with `pip install faster-whisper torch`."
            )

        if self._model is None:
            print(f"Loading Whisper {self.model_size} model (first time may download)...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            print("Model loaded!")

    def _run_stream(self, on_transcript: Callable[[TranscriptEvent], None], emit_partials: bool) -> None:
        """Main streaming loop that captures audio and processes with Whisper."""
        assert sd is not None
        assert self._model is not None

        chunk_samples = int(self.sample_rate * self.chunk_duration)
        overlap_samples = int(self.sample_rate * self.overlap)
        audio_buffer: list[float] = []

        def audio_callback(indata, frames, time_info, status):  # pragma: no cover
            if status:
                print(f"[Audio warning: {status}]")
            # Convert to mono float32
            mono = indata[:, 0] if indata.ndim > 1 else indata
            self._audio_queue.put(mono.copy())
            # Forward raw audio for speaker ID if callback provided
            if self._on_audio_chunk:
                import numpy as np
                # Convert float32 to int16 bytes for compatibility
                int16_audio = (mono * 32767).astype(np.int16)
                self._on_audio_chunk(int16_audio.tobytes(), self.sample_rate)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            blocksize=chunk_samples // 4,  # Small blocks for responsiveness
        )

        with stream:
            import numpy as np
            
            # Buffer for accumulating audio until silence is detected
            phrase_buffer: list[float] = []
            last_speech_timestamp: Optional[float] = None
            last_partial_timestamp: Optional[float] = None
            partial_update_interval = 1.0  # Emit partial transcripts every 1 second during speech
            
            while not self._stop_event.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check if we should process due to silence
                    if phrase_buffer and last_speech_timestamp is not None:
                        time_since_speech = time.time() - last_speech_timestamp
                        if time_since_speech >= self.silence_duration:
                            # Process accumulated phrase after silence
                            if len(phrase_buffer) >= self.sample_rate * 0.3:  # At least 0.3 seconds
                                audio_array = np.array(phrase_buffer, dtype=np.float32)
                                
                                # Print status before transcribing
                                print("\n[STATUS] Transcribing...")
                                
                                # Transcribe with Whisper
                                segments, _ = self._model.transcribe(
                                    audio_array,
                                    language=self.language,
                                    beam_size=self.beam_size,
                                    vad_filter=True,
                                    vad_parameters=dict(min_silence_duration_ms=500),
                                )
                                
                                # Collect segments
                                text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
                                if text_parts:
                                    full_text = " ".join(text_parts)
                                    on_transcript(TranscriptEvent(text=full_text, is_final=True))
                                
                                # Clear buffer
                                phrase_buffer.clear()
                                last_speech_timestamp = None
                                last_partial_timestamp = None
                    # Check if we should emit partial transcript during active speech
                    elif emit_partials and phrase_buffer and last_speech_timestamp is not None:
                        time_since_partial = time.time() - (last_partial_timestamp or last_speech_timestamp)
                        if time_since_partial >= partial_update_interval and len(phrase_buffer) >= self.sample_rate * 0.5:
                            # Emit partial transcript
                            audio_array = np.array(phrase_buffer, dtype=np.float32)
                            segments, _ = self._model.transcribe(
                                audio_array,
                                language=self.language,
                                beam_size=self.beam_size,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=500),
                            )
                            text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
                            if text_parts:
                                partial_text = " ".join(text_parts)
                                on_transcript(TranscriptEvent(text=partial_text, is_final=False))
                            last_partial_timestamp = time.time()
                    continue

                if len(chunk) == 0:
                    continue

                # Check if this chunk contains speech (above threshold)
                chunk_array = np.array(chunk)
                max_amplitude = np.abs(chunk_array).max()
                has_speech = max_amplitude > self.silence_threshold
                
                # Always add chunk to buffer
                phrase_buffer.extend(chunk)
                
                if has_speech:
                    # Update timestamp of last speech
                    last_speech_timestamp = time.time()
                else:
                    # Silence detected - check if we've had enough silence
                    if phrase_buffer and last_speech_timestamp is not None:
                        time_since_speech = time.time() - last_speech_timestamp
                        if time_since_speech >= self.silence_duration:
                            # Process accumulated phrase
                            if len(phrase_buffer) >= self.sample_rate * 0.3:  # At least 0.3 seconds
                                audio_array = np.array(phrase_buffer, dtype=np.float32)
                                
                                # Print status before transcribing
                                print("\n[STATUS] Transcribing...")
                                
                                # Transcribe with Whisper
                                segments, _ = self._model.transcribe(
                                    audio_array,
                                    language=self.language,
                                    beam_size=self.beam_size,
                                    vad_filter=True,
                                    vad_parameters=dict(min_silence_duration_ms=500),
                                )
                                
                                # Collect segments
                                text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
                                if text_parts:
                                    full_text = " ".join(text_parts)
                                    on_transcript(TranscriptEvent(text=full_text, is_final=True))
                                
                                # Clear buffer
                                phrase_buffer.clear()
                                last_speech_timestamp = None

        # Process remaining audio in phrase buffer
        if phrase_buffer and not self._stop_event.is_set():
            import numpy as np

            audio_array = np.array(phrase_buffer, dtype=np.float32)
            if len(audio_array) > self.sample_rate * 0.3:  # At least 0.3 seconds
                # Print status before transcribing
                print("\n[STATUS] Transcribing...")
                
                segments, _ = self._model.transcribe(
                    audio_array,
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=True,
                )
                text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
                if text_parts:
                    full_text = " ".join(text_parts)
                    on_transcript(TranscriptEvent(text=full_text, is_final=True))


def backend_available() -> bool:
    """Return True when both Whisper and sounddevice dependencies are installed."""
    return sd is not None and WhisperModel is not None


def stream_live(
    on_transcript: Callable[[TranscriptEvent], None],
    **kwargs,
) -> WhisperLiveStreamer:
    """
    Convenience helper that instantiates ``WhisperLiveStreamer``, starts it, and
    returns the instance so the caller can later invoke ``stop()``.
    """
    streamer = WhisperLiveStreamer(**kwargs)
    streamer.start(on_transcript)
    return streamer

