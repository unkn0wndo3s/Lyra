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

    def __init__(self, recognizer: Optional[Any] = None, microphone_class: Optional[Any] = None) -> None:
        if sr is None:
            raise RuntimeError("speech_recognition is not installed. Install it to use SpeechInputProvider.")
        self.recognizer = recognizer or sr.Recognizer()
        self.microphone_class = microphone_class or sr.Microphone

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
            if engine == "google":
                text = self.recognizer.recognize_google(audio, language=language)
            elif engine == "sphinx":
                text = self.recognizer.recognize_sphinx(audio, language=language)
            else:
                raise ValueError(f"Unsupported engine: {engine}")
        except Exception as exc:  # noqa: BLE001
            raise InputCaptureError(f"Speech transcription failed: {exc}") from exc
        metadata = {"language": language, "engine": engine}
        return self._wrap(text, metadata)


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


