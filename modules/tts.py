"""
Text-to-speech (TTS) utilities.

The module exposes a ``TTSEngine`` class that converts text to speech and plays it
using the system audio output. It supports multiple backends:

- gTTS (Google Text-to-Speech): Requires internet connection, higher quality
- pyttsx3: Offline, uses system TTS engine

The implementation depends on optional third-party packages:

    pip install gtts librosa soundfile  # For gTTS backend
    pip install pyttsx3                 # For offline backend
    pip install sounddevice             # Required for audio playback
"""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Literal, Optional

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from gtts import gTTS  # type: ignore
    import librosa  # type: ignore
    GTTS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    GTTS_AVAILABLE = False
    gTTS = None  # type: ignore
    librosa = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pyttsx3  # type: ignore
    PYTTSX3_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None  # type: ignore


class BackendUnavailable(RuntimeError):
    """Raised when the TTS backend dependencies are missing."""


class TTSEngine:
    """
    Text-to-speech engine that converts text to audio and plays it.

    Example usage::

        from modules.tts import TTSEngine

        # Use gTTS (requires internet)
        tts = TTSEngine(backend="gtts")
        tts.speak("Hello, world!")

        # Use pyttsx3 (offline)
        tts = TTSEngine(backend="pyttsx3")
        tts.speak("Hello, world!")
    """

    def __init__(
        self,
        *,
        backend: Literal["gtts", "pyttsx3"] = "gtts",
        sample_rate: int = 22050,
        lang: str = "en",
    ) -> None:
        """
        Initialize the TTS engine.

        Args:
            backend: TTS backend to use ("gtts" or "pyttsx3")
            sample_rate: Audio sample rate (for gTTS backend)
            lang: Language code (for gTTS backend, e.g., "en", "fr")
        """
        if sd is None:
            raise BackendUnavailable(
                "TTS requires the 'sounddevice' package. "
                "Install it with `pip install sounddevice`."
            )

        self.backend = backend
        self.sample_rate = sample_rate
        self.lang = lang
        self._tts_lock = threading.Lock()  # Lock for pyttsx3 to prevent concurrent calls

        if backend == "gtts":
            if not GTTS_AVAILABLE:
                raise BackendUnavailable(
                    "gTTS backend requires 'gtts' and 'librosa' packages. "
                    "Install them with `pip install gtts librosa soundfile`."
                )
        elif backend == "pyttsx3":
            if not PYTTSX3_AVAILABLE:
                raise BackendUnavailable(
                    "pyttsx3 backend requires the 'pyttsx3' package. "
                    "Install it with `pip install pyttsx3`."
                )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'gtts' or 'pyttsx3'.")

    def speak(self, text: str) -> None:
        """
        Speak the given text using the configured backend.

        Args:
            text: Text to speak
        """
        if not text.strip():
            return

        if self.backend == "gtts":
            self._speak_gtts(text)
        elif self.backend == "pyttsx3":
            self._speak_pyttsx3(text)

    def _speak_gtts(self, text: str) -> None:
        """Speak using gTTS backend."""
        assert gTTS is not None and librosa is not None and sd is not None  # type hints

        try:
            # Generate audio with gTTS
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tts = gTTS(text=text, lang=self.lang, slow=False)
                tts.save(tmp.name)
                tmp_path = tmp.name

            try:
                # Load audio file using librosa
                audio_data, sr = librosa.load(tmp_path, sr=self.sample_rate)
                
                # Play audio using sounddevice
                sd.play(audio_data, samplerate=sr)
                sd.wait()  # Wait until playback is finished
            finally:
                # Cleanup
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            raise RuntimeError(f"gTTS error: {e}") from e

    def _speak_pyttsx3(self, text: str) -> None:
        """Speak using pyttsx3 backend."""
        assert pyttsx3 is not None  # type hint

        # pyttsx3: Use lock to prevent concurrent calls and create fresh engine each time
        # This avoids "run loop already started" errors when called from multiple threads
        with self._tts_lock:
            try:
                # Create a fresh engine instance for this call
                # pyttsx3 can't reuse engines after runAndWait(), so we create new ones
                engine = pyttsx3.init()
                
                # Configure voice settings
                voices = engine.getProperty('voices')
                if voices:
                    # Try to find a female voice, otherwise use the first available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                    else:
                        engine.setProperty('voice', voices[0].id)
                
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                # Use the fresh engine for this call
                engine.say(text)
                engine.runAndWait()
                # Engine is automatically cleaned up after runAndWait() completes
            except Exception as e:
                raise RuntimeError(f"pyttsx3 error: {e}") from e


def backend_available(backend: Literal["gtts", "pyttsx3"] = "gtts") -> bool:
    """
    Return True when the specified TTS backend dependencies are installed.

    Args:
        backend: Backend to check ("gtts" or "pyttsx3")
    """
    if sd is None:
        return False
    
    if backend == "gtts":
        return GTTS_AVAILABLE
    elif backend == "pyttsx3":
        return PYTTSX3_AVAILABLE
    else:
        return False


def create_tts_engine(
    backend: Optional[Literal["gtts", "pyttsx3"]] = None,
    **kwargs,
) -> Optional[TTSEngine]:
    """
    Convenience helper that creates a TTSEngine with automatic backend selection.

    Tries gTTS first, falls back to pyttsx3 if gTTS is not available.

    Args:
        backend: Backend to use (None for auto-selection)
        **kwargs: Additional arguments passed to TTSEngine

    Returns:
        TTSEngine instance, or None if no backend is available
    """
    if backend is None:
        # Auto-select: try gTTS first, then pyttsx3
        if backend_available("gtts"):
            backend = "gtts"
        elif backend_available("pyttsx3"):
            backend = "pyttsx3"
        else:
            return None
    
    try:
        return TTSEngine(backend=backend, **kwargs)
    except (BackendUnavailable, ValueError):
        return None

