"""
Live speech-to-text streaming utilities.

The module exposes a ``LiveSpeechStreamer`` class that captures audio from the
system microphone and produces incremental transcripts using the Vosk speech
recognition engine. Partial words are emitted while the speaker is still talking
so clients can react immediately.

The implementation depends on optional third-party packages:

    pip install vosk sounddevice

You must also download a Vosk acoustic model and point the streamer to it via
``model_path`` or the ``VOSK_MODEL_PATH`` environment variable.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import vosk  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    vosk = None  # type: ignore


@dataclass(frozen=True)
class TranscriptEvent:
    """Represents a chunk of recognized speech."""

    text: str
    is_final: bool


class BackendUnavailable(RuntimeError):
    """Raised when the speech backend dependencies are missing."""


class LiveSpeechStreamer:
    """
    Stream audio from the default microphone and emit transcripts in real time.

    Example usage::

        from modules.live_speech import LiveSpeechStreamer

        def on_event(event):
            stamp = "FINAL" if event.is_final else "LIVE"
            print(stamp, event.text)

        streamer = LiveSpeechStreamer(model_path="models/vosk-model-small-en-us-0.15")
        streamer.start(on_event)
        # ... later ...
        streamer.stop()
    """

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        sample_rate: int = 16_000,
        device: Optional[int | str] = None,
        block_duration: float = 0.5,
    ) -> None:
        self.model_path = model_path or os.environ.get("VOSK_MODEL_PATH")
        self.sample_rate = sample_rate
        self.device = device
        self.block_duration = block_duration

        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # --------------------------------------------------------------------- API
    def start(
        self,
        on_transcript: Callable[[TranscriptEvent], None],
        *,
        emit_partials: bool = True,
    ) -> None:
        """Begin streaming and invoke ``on_transcript`` as words are recognized."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("LiveSpeechStreamer is already running.")

        recognizer = self._prepare_backend()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_stream,
            args=(recognizer, on_transcript, emit_partials),
            daemon=True,
            name="LiveSpeechStreamer",
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Stop streaming and wait for the worker thread to finish."""
        if not self._thread:
            return
        self._stop_event.set()
        self._queue.put_nowait(b"")  # unblock the queue
        self._thread.join(timeout=timeout)
        self._thread = None
        self._stop_event.clear()

    def __enter__(self) -> "LiveSpeechStreamer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ----------------------------------------------------------------- Helpers
    def _prepare_backend(self):
        if vosk is None or sd is None:
            raise BackendUnavailable(
                "Streaming requires the 'vosk' and 'sounddevice' packages. "
                "Install them with `pip install vosk sounddevice`."
            )
        if not self.model_path:
            raise BackendUnavailable(
                "A Vosk model path is required. Provide the `model_path` argument "
                "or set the VOSK_MODEL_PATH environment variable."
            )
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Vosk model not found at {model_dir}")

        model = vosk.Model(str(model_dir))
        recognizer = vosk.KaldiRecognizer(model, self.sample_rate)
        recognizer.SetWords(True)
        return recognizer

    def _run_stream(self, recognizer, on_transcript, emit_partials):
        assert sd is not None  # mypy hint

        def audio_callback(indata, frames, time, status):  # pragma: no cover - realtime
            if status:
                # Drop status into queue as empty chunk so loop can report it later.
                self._queue.put(status)
            self._queue.put(bytes(indata))

        blocksize = int(self.sample_rate * self.block_duration)
        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=blocksize,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        )

        with stream:  # pragma: no cover - realtime
            while not self._stop_event.is_set():
                try:
                    data = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if isinstance(data, sd.CallbackFlags):
                    # Surface warnings to the caller as partial text events.
                    on_transcript(
                        TranscriptEvent(
                            text=f"[audio warning: {data}]",
                            is_final=False,
                        )
                    )
                    continue

                if not data:
                    continue

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        on_transcript(TranscriptEvent(text=text, is_final=True))
                elif emit_partials:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial:
                        on_transcript(TranscriptEvent(text=partial, is_final=False))


def backend_available() -> bool:
    """Return True when both Vosk and sounddevice dependencies are installed."""
    return vosk is not None and sd is not None


def stream_live(
    on_transcript: Callable[[TranscriptEvent], None],
    **kwargs,
) -> LiveSpeechStreamer:
    """
    Convenience helper that instantiates ``LiveSpeechStreamer``, starts it, and
    returns the instance so the caller can later invoke ``stop()``.
    """
    streamer = LiveSpeechStreamer(**kwargs)
    streamer.start(on_transcript)
    return streamer

