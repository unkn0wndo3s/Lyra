"""Coordinator for Lyra input providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import InputResult, InputType
from .providers import (
    ExternalSensorConfiguration,
    ExternalSensorProvider,
    ImageInputProvider,
    OPENCV_AVAILABLE,
    SpeechInputProvider,
    SPEECH_RECOGNITION_AVAILABLE,
    TextInputProvider,
)


class InputManager:
    """Facade to capture inputs from multiple modalities."""

    def __init__(
        self,
        *,
        text_provider: Optional[TextInputProvider] = None,
        speech_provider: Optional[SpeechInputProvider] = None,
        image_provider: Optional[ImageInputProvider] = None,
        sensor_provider: Optional[ExternalSensorProvider] = None,
    ) -> None:
        self.text_provider = text_provider or TextInputProvider()
        self.speech_provider = speech_provider
        self.image_provider = image_provider
        self.sensor_provider = sensor_provider or ExternalSensorProvider()

    def ensure_speech_provider(self) -> SpeechInputProvider:
        if not self.speech_provider:
            self.speech_provider = SpeechInputProvider()
        return self.speech_provider

    def ensure_image_provider(self) -> ImageInputProvider:
        if not self.image_provider:
            self.image_provider = ImageInputProvider()
        return self.image_provider

    def add_text(self, *, text: Optional[str] = None, prompt: Optional[str] = None, input_fn: Callable[[str], str] = input) -> InputResult:
        return self.text_provider.capture(text=text, prompt=prompt, input_fn=input_fn)

    def transcribe_speech(
        self,
        *,
        from_microphone: bool = True,
        file_path: Optional[str | Path] = None,
        language: str = "en-US",
        engine: str = "google",
        timeout: int = 5,
        phrase_time_limit: Optional[int] = 15,
    ) -> InputResult:
        provider = self.ensure_speech_provider()
        if from_microphone:
            return provider.transcribe_from_microphone(
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
                language=language,
                engine=engine,
            )
        if not file_path:
            raise ValueError("file_path is required when from_microphone is False.")
        return provider.transcribe_from_file(
            file_path=file_path,
            language=language,
            engine=engine,
        )

    def capture_image(
        self,
        *,
        file_path: str | Path,
        color_mode: int = 1,
        preprocess: Optional[Callable[[Any], Any]] = None,
    ) -> InputResult:
        provider = self.ensure_image_provider()
        return provider.capture_from_file(
            file_path=file_path,
            color_mode=color_mode,
            preprocess=preprocess,
        )

    def register_sensor(self, *, name: str, capture_fn: Callable[[], Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        config = ExternalSensorConfiguration(name=name, capture_fn=capture_fn, metadata=metadata)
        self.sensor_provider.register_sensor(config)

    def capture_sensor(self, name: str) -> InputResult:
        return self.sensor_provider.capture(name)

    def available_modalities(self) -> List[InputType]:
        modalities = [InputType.TEXT, InputType.SENSOR]
        if self.speech_provider or SPEECH_RECOGNITION_AVAILABLE:
            modalities.append(InputType.SPEECH)
        if self.image_provider or OPENCV_AVAILABLE:
            modalities.append(InputType.IMAGE)
        return modalities


