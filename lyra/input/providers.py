"""Concrete input providers for text, speech, image, and sensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import speech_recognition as sr  # type: ignore
except ImportError:  # pragma: no cover
    sr = None

OPENCV_AVAILABLE = cv2 is not None
SPEECH_RECOGNITION_AVAILABLE = sr is not None

from .models import InputCaptureError, InputResult, InputType


@dataclass
class SpeechStreamHandle:
    """Controller returned by continuous listening sessions."""

    stop_callable: Callable[[bool], None]
    microphone: Any
    callback: Callable[[InputResult], None]
    active: bool = True

    def stop(self, *, wait_for_stop: bool = True) -> None:
        if not self.active:
            return
        self.stop_callable(wait_for_stop)
        self.active = False

    def __enter__(self) -> "SpeechStreamHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(wait_for_stop=False)


class BaseInputProvider:
    """Common utilities for all providers."""

    input_type: InputType

    def _wrap(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> InputResult:
        return InputResult(type=self.input_type, content=content, metadata=metadata or {})


class TextInputProvider(BaseInputProvider):
    """Simple text input handler."""

    input_type = InputType.TEXT

    def capture(self, *, text: Optional[str] = None, prompt: Optional[str] = None, input_fn: Callable[[str], str] = input) -> InputResult:
        """Capture text either from a provided string or via stdin."""
        if text is None:
            if not prompt:
                prompt = "Enter input: "
            text = input_fn(prompt)
        metadata = {"length": len(text)}
        return self._wrap(text, metadata)


class SpeechInputProvider(BaseInputProvider):
    """Speech-to-text handler built on speech_recognition."""

    input_type = InputType.SPEECH

    def __init__(
        self,
        recognizer: Optional[Any] = None,
        microphone_class: Optional[Any] = None,
        *,
        energy_threshold: Optional[int] = None,
        dynamic_energy_threshold: bool = True,
        pause_threshold: float = 0.8,
    ) -> None:
        if sr is None:
            raise RuntimeError("speech_recognition is not installed. Install it to use SpeechInputProvider.")
        self.recognizer = recognizer or sr.Recognizer()
        self.microphone_class = microphone_class or sr.Microphone
        if energy_threshold is not None:
            self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold
        self.recognizer.pause_threshold = pause_threshold

    def transcribe_from_microphone(
        self,
        *,
        timeout: int = 5,
        phrase_time_limit: Optional[int] = 15,
        language: str = "en-US",
        engine: str = "google",
    ) -> InputResult:
        """Capture audio from the default microphone and transcribe it."""
        with self.microphone_class() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        return self._transcribe_audio(audio, language=language, engine=engine)

    def transcribe_from_file(
        self,
        file_path: str | Path,
        *,
        language: str = "en-US",
        engine: str = "google",
    ) -> InputResult:
        """Transcribe a prerecorded audio file."""
        audio_path = Path(file_path)
        if not audio_path.exists():
            raise InputCaptureError(f"Audio file does not exist: {audio_path}")
        with sr.AudioFile(str(audio_path)) as source:
            audio = self.recognizer.record(source)
        return self._transcribe_audio(audio, language=language, engine=engine)

    def _transcribe_audio(self, audio: Any, *, language: str, engine: str) -> InputResult:
        try:
            text = self._recognize_with_engine(audio, engine, language)
        except Exception as exc:  # noqa: BLE001
            raise InputCaptureError(f"Speech transcription failed: {exc}") from exc
        metadata = {"language": language, "engine": engine}
        return self._wrap(text, metadata)

    def _recognize_with_engine(self, audio: Any, engine: str, language: str) -> str:
        engine = engine.lower()
        if engine == "google":
            return self.recognizer.recognize_google(audio, language=language)
        if engine == "sphinx":
            return self.recognizer.recognize_sphinx(audio, language=language)
        raise ValueError(f"Unsupported engine: {engine}")

    def start_continuous_listening(
        self,
        callback: Callable[[InputResult], None],
        *,
        languages: Optional[List[str]] = None,
        engines: Optional[List[str]] = None,
        phrase_time_limit: Optional[int] = None,
        ambient_duration: float = 1.0,
        error_handler: Optional[Callable[[Exception], None]] = None,
    ) -> SpeechStreamHandle:
        """Begin continuous listening in a background thread.

        The callback receives InputResult objects for each successful transcription.
        """

        if sr is None:
            raise RuntimeError("speech_recognition is not installed.")
        if callback is None:
            raise ValueError("callback must be provided for continuous listening.")

        languages = languages or ["en-US"]
        engines = engines or ["google"]

        microphone = self.microphone_class()
        with microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=ambient_duration)

        def _background(recognizer: Any, audio: Any) -> None:
            last_error: Optional[Exception] = None
            for language in languages:
                for engine in engines:
                    try:
                        text = self._recognize_with_engine(audio, engine, language)
                    except sr.UnknownValueError as exc:  # type: ignore[attr-defined]
                        last_error = exc
                        continue
                    except sr.RequestError as exc:  # type: ignore[attr-defined]
                        last_error = exc
                        continue
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        continue
                    else:
                        metadata = {"language": language, "engine": engine}
                        callback(self._wrap(text, metadata))
                        return
            if error_handler and last_error:
                error_handler(last_error)

        stop_callable = self.recognizer.listen_in_background(
            microphone, _background, phrase_time_limit=phrase_time_limit
        )
        return SpeechStreamHandle(stop_callable=stop_callable, microphone=microphone, callback=callback)


class ImageInputProvider(BaseInputProvider):
    """Image ingestion via OpenCV."""

    input_type = InputType.IMAGE

    def __init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed. Install it to use ImageInputProvider.")

    def capture_from_file(
        self,
        file_path: str | Path,
        *,
        color_mode: int = 1,  # 1 = BGR, 0 = grayscale
        preprocess: Optional[Callable[[Any], Any]] = None,
    ) -> InputResult:
        image_path = Path(file_path)
        if not image_path.exists():
            raise InputCaptureError(f"Image file does not exist: {image_path}")
        image = cv2.imread(str(image_path), color_mode)
        if image is None:
            raise InputCaptureError(f"OpenCV could not load image: {image_path}")
        if preprocess:
            image = preprocess(image)
        metadata = {"shape": image.shape, "color_mode": color_mode, "path": str(image_path)}
        return self._wrap(image, metadata)


@dataclass
class ExternalSensorConfiguration:
    """Describes an external sensor callback."""

    name: str
    capture_fn: Callable[[], Any]
    metadata: Optional[Dict[str, Any]] = None


class ExternalSensorProvider(BaseInputProvider):
    """Generic adapter for arbitrary sensors or device feeds."""

    input_type = InputType.SENSOR

    def __init__(self, sensors: Optional[List[ExternalSensorConfiguration]] = None) -> None:
        self.sensors = {sensor.name: sensor for sensor in sensors or []}

    def register_sensor(self, config: ExternalSensorConfiguration) -> None:
        self.sensors[config.name] = config

    def capture(self, name: str) -> InputResult:
        if name not in self.sensors:
            raise InputCaptureError(f"Sensor '{name}' has not been registered.")
        config = self.sensors[name]
        try:
            data = config.capture_fn()
        except Exception as exc:  # noqa: BLE001
            raise InputCaptureError(f"Sensor '{name}' capture failed: {exc}") from exc
        metadata = dict(config.metadata or {})
        metadata["sensor"] = name
        return self._wrap(data, metadata)


