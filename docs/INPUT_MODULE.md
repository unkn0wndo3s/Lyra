# Data Input Module

Lyra's input subsystem unifies multiple modalities—text, speech, images, and
external sensors—behind a single `InputManager`.

## Core Types

- `InputType`: enum of supported channels (`TEXT`, `SPEECH`, `IMAGE`, `SENSOR`).
- `InputResult`: normalized payload containing the captured content plus
  metadata (e.g., transcription language, image shape).
- `InputCaptureError`: raised when a provider fails to acquire or decode data.

## Providers

| Provider | Description | Key Methods |
|----------|-------------|-------------|
| `TextInputProvider` | Console/CLI text capture. | `capture(text=None, prompt=None)` |
| `SpeechInputProvider` | Speech-to-text via `speech_recognition`. | `transcribe_from_microphone(...)`, `transcribe_from_file(...)` |
| `ImageInputProvider` | Image ingestion with OpenCV. | `capture_from_file(path, color_mode=1)` |
| `ExternalSensorProvider` | Generic adapter for hardware/sensor callbacks. | `register_sensor`, `capture(name)` |

## Input Manager

Instantiate `InputManager` to orchestrate providers:

```python
from lyra import InputManager, InputType

inputs = InputManager()
text = inputs.add_text(text="status report")
speech = inputs.transcribe_speech(from_microphone=False, file_path="sample.wav")
image = inputs.capture_image(file_path="frame.png")

def read_temperature():
    # replace with actual sensor call
    return 22.5

inputs.register_sensor(name="thermometer", capture_fn=read_temperature)
sensor = inputs.capture_sensor("thermometer")
```

The manager lazily instantiates speech and image providers so deployments that
do not install `speech_recognition` or `opencv-python` can still work with text
inputs and sensors.

## Garbage Collection & External Devices

The module delegates long-lived memory cleanup to `MemoryInterface`, but sensor
feeds can register arbitrary metadata to support downstream filtering. If an
external device becomes unavailable, its callback should raise an exception,
which the provider surfaces as `InputCaptureError`.


