# Speech Recognition

Lyra relies on the `speech_recognition` package for microphone capture and
transcription. The `SpeechInputProvider` exposes both one-off transcription
helpers and a continuous listening API that can stream results into callbacks.

## Prerequisites

Install the required dependencies:

```bash
sudo apt install portaudio19-dev python3-dev  # OS packages
./venv/bin/pip install pyaudio                # Python binding needed by speech_recognition
```

## One-Off Transcription

```python
from lyra import InputManager, InputType

inputs = InputManager()
text = inputs.transcribe_speech(from_microphone=True, language="en-US")
print(text.content)
```

## Continuous Listening

```python
from lyra import InputManager, SpeechStreamHandle

provider = InputManager().ensure_speech_provider()  # or SpeechInputProvider()

def on_transcript(result):
    print("heard:", result.content, result.metadata)

handle: SpeechStreamHandle = provider.start_continuous_listening(
    callback=on_transcript,
    languages=["en-US", "en-GB"],   # try multiple accents
    engines=["google", "sphinx"],   # fall back if the first engine fails
    phrase_time_limit=5,
)

try:
    while True:
        ...
finally:
    handle.stop()
```

Features:

- **Ambient noise calibration** (`adjust_for_ambient_noise`) keeps accuracy high
  in changing environments.
- **Multiple languages/engines** allow better handling of different speech
  patterns or accents. The listener iterates through each combination until a
  result is produced.
- **Error callbacks** via `error_handler` surface repeated failures (e.g.,
  connectivity issues, unintelligible audio) without crashing the main loop.

## Integration with Real-Time Processor

Use `start_continuous_listening` inside a background task and forward each
transcript to the `RealTimeProcessor` queue, or register a speech source that
pulls from pre-recorded files when running automated tests.


