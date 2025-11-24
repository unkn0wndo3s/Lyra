# Module Loader

This document explains how the dynamic loader in `module_loader.py` discovers,
loads, and reloads modules that live under the local `modules/` package.

## Overview

- Automatically imports every module inside `modules/` as soon as
  `module_loader.py` is imported, so common helpers are immediately available.
- Keeps track of all classes and functions defined in those modules to simplify
  inspection and debugging.
- Provides utility functions to reload all modules, reload a single module, or
  load any modules that were added after startup without restarting the process.

## Directory Layout

```
Lyra/
├── module_loader.py        # Dynamic loader script
└── modules/
    ├── __init__.py         # Marks directory as a package
    ├── live_speech.py      # Live speech-to-text streamer (Vosk + sounddevice)
    ├── emotion_interpreter.py  # Heuristic emotion detection for transcripts
    ├── voice_identity.py   # Voice sampling and recognition utilities
    ├── realtime_monitor.py # Fusion of STT + emotions + speaker ID
    ├── mouse_controller.py # Non-linear mouse automation helpers
    ├── keyboard_controller.py # Human-like keyboard automation
    ├── text_recognition.py # Advanced OCR helpers (PaddleOCR)
    ├── video_text_recognition.py # Video OCR with temporal voting
    └── morpion_solver.py # Optimal tic-tac-toe (morpion) solver
```

You can drop any number of additional `.py` files into `modules/`; the loader
discovers them automatically.

## Default Loading Behavior

Importing `module_loader.py` (or executing it directly) triggers `_default_load`
which calls `load_all_modules()` to import every module reported by
`discover_module_names()`. Each module is wrapped in a `LoadedModule` record
containing:

- `module`: actual module object.
- `classes`: dict of class names → class objects defined in that module.
- `functions`: dict of function names → function objects defined in that module.

## API Surface

The loader exposes four main helpers for runtime control:

| Function | Description |
| --- | --- |
| `load_all_modules()` | Imports every module detected under `modules/`. |
| `reload_all_modules()` | Calls `importlib.reload` on every module that has already been loaded. |
| `reload_module(name)` | Reloads a single named module; raises `KeyError` if it was never loaded. |
| `load_unloaded_modules()` | Imports only the modules that have been added since the last load. |
| `get_namespace()` | Produces a merged dictionary of classes/functions drawn from multiple modules. |

Each helper updates the global `LOADED_MODULES` dictionary so you can inspect
what is currently available (for example by calling
`summarize_loaded_modules(LOADED_MODULES.items())`).

### Using Exports from Multiple Modules

`get_namespace()` lets you pull functions and classes from several modules into
one dictionary so they can be used side-by-side without manual lookups:

```python
from module_loader import get_namespace

namespace = get_namespace()  # defaults to all modules, qualified names
stream = namespace["live_speech.LiveSpeechStreamer"]
emotion_interpret = namespace["emotion_interpreter.interpret_text"]
voice_identifier_cls = namespace["voice_identity.VoiceIdentifier"]
insight_session = namespace["realtime_monitor.start_insight_session"]
mouse_move = namespace["mouse_controller.move_mouse"]
keyboard = namespace["keyboard_controller.default_keyboard"]
ocr_extract = namespace["text_recognition.extract_text"]
video_ocr = namespace["video_text_recognition.recognize_video_text"]
morpion = namespace["morpion_solver.best_move"]

streamer = stream(model_path="/path/to/vosk-model")
emotion = emotion_interpret("I am thrilled to be here!")
print(emotion.formatted)  # this is **happy** functionning
identifier = voice_identifier_cls()
print(identifier.known_speakers)  # ()
# launch full realtime session (requires optional deps)
session = insight_session()
mouse_move(200, 300)
keyboard().space()
print(ocr_extract("poster.png"))
video_result = video_ocr("ad_clip.mp4")
print(video_result.dominant_text, video_result.candidates)
print(morpion(["X O", "   ", "   "]))  # -> (0, 2)
```

Pass a list of module names to limit the selection, disable `qualified_names`
for raw symbol names (will raise on collisions), or set
`include_classes=False`/`include_functions=False` to filter what gets merged.

## Running the Script

To run the loader directly and view a human-readable summary:

```bash
python module_loader.py
```

The script prints the modules loaded on startup, reloads them in-place, and then
reports if any new modules were discovered.

## Adding New Modules

1. Create a new file inside `modules/`, e.g. `modules/image_utils.py`.
2. Define any classes or functions you want exposed.
3. Run `python module_loader.py` or import `module_loader` inside your program.
4. Call `load_unloaded_modules()` to import the new file without restarting.

If you change existing modules, call `reload_module("emotion_interpreter")` or
`reload_all_modules()` to pick up the updates at runtime.

## Live Speech Module

`modules/live_speech.py` provides `LiveSpeechStreamer`, which emits partial and
final transcripts while the speaker is still talking. It uses the Vosk engine
with a default larger model (vosk-model-en-us-0.22) for better accuracy. The
`sounddevice` library is required:

```bash
pip install vosk sounddevice
```

A default Vosk model is bundled in `models/`, so you can start immediately:

```python
from modules.live_speech import LiveSpeechStreamer

def on_transcript(event):
    print("FINAL" if event.is_final else "LIVE", event.text)

streamer = LiveSpeechStreamer()  # Uses default model automatically
streamer.start(on_transcript)
```

For **maximum accuracy**, use the Whisper-based streamer instead (see below).

**Run command:**
```bash
python -c "from modules.live_speech import LiveSpeechStreamer; import time; streamer = LiveSpeechStreamer(); print('Starting live speech recognition (requires microphone, press Ctrl+C to stop)...'); streamer.start(lambda e: print('FINAL' if e.is_final else 'LIVE:', e.text)); exec('while True: time.sleep(1)')"
```

Note: This requires a microphone. Press Ctrl+C to stop the recognition.

## Whisper Live Speech Module (High Accuracy)

`modules/whisper_live.py` provides `WhisperLiveStreamer`, which uses OpenAI's
Whisper large-v3 model for superior transcription accuracy compared to Vosk.
The streamer accumulates audio and waits for you to stop speaking (detects silence)
before generating the transcription, providing more accurate results.

Install dependencies:

```bash
pip install faster-whisper sounddevice torch
```

The model (~3GB) downloads automatically on first use. Usage:

```python
from modules.whisper_live import WhisperLiveStreamer

def on_transcript(event):
    print("FINAL" if event.is_final else "LIVE", event.text)

streamer = WhisperLiveStreamer(
    model_size="large-v3",  # Best accuracy: large-v3 or large-v3-turbo (faster, similar accuracy)
    # Other options: tiny, base, small, medium, large-v2
    device="cpu",  # Use "cuda" for GPU acceleration
    compute_type="int8",  # int8/int16/float16/float32
    silence_duration=1.5,  # Seconds of silence before processing (default: 1.5)
    silence_threshold=0.01,  # Audio amplitude threshold for silence (default: 0.01)
)
streamer.start(on_transcript)
```

Whisper is significantly more accurate than Vosk, especially for:
- Accented speech
- Background noise
- Multiple speakers
- Technical terminology
- Non-standard pronunciations

**Model Comparison:**
- **large-v3-turbo** (recommended): Fastest large model, similar accuracy to large-v3, optimized decoder
- **large-v3**: Best accuracy, slower than turbo version (~3GB download)
- **large-v2**: High accuracy, slightly older (~1.5GB download)
- **medium**: Good balance of speed and accuracy (~769MB download)
- **small**: Fast with decent accuracy (~244MB download)
- **base**: Fast, good for testing (~74MB download)
- **tiny**: Fastest but lowest accuracy (~39MB download)

For production use, `large-v3-turbo` offers the best speed/accuracy trade-off.

**Run command (with large-v3-turbo for best speed/accuracy balance):**
```bash
python -c "from modules.whisper_live import WhisperLiveStreamer; import time; streamer = WhisperLiveStreamer(model_size='large-v3-turbo', device='cpu', compute_type='int8'); print('Starting Whisper live recognition (requires microphone, model downloads on first use, press Ctrl+C to stop)...'); streamer.start(lambda e: print('FINAL' if e.is_final else 'LIVE:', e.text)); exec('while True: time.sleep(1)')"
```

**Run command (with large-v3 for maximum accuracy):**
```bash
python -c "from modules.whisper_live import WhisperLiveStreamer; import time; streamer = WhisperLiveStreamer(model_size='large-v3', device='cpu', compute_type='int8'); print('Starting Whisper live recognition (requires microphone, model downloads on first use, press Ctrl+C to stop)...'); streamer.start(lambda e: print('FINAL' if e.is_final else 'LIVE:', e.text)); exec('while True: time.sleep(1)')"
```

Note: This requires a microphone. Model sizes and download sizes:
- `tiny`: ~39MB, fastest, lowest accuracy
- `base`: ~74MB, fast, good for testing
- `small`: ~244MB, balanced
- `medium`: ~769MB, good accuracy
- `large-v2`: ~1.5GB, high accuracy
- `large-v3`: ~3GB, best accuracy (default)
- `large-v3-turbo`: ~3GB, faster than large-v3 with similar accuracy (recommended for speed)

Press Ctrl+C to stop the recognition.

## Emotion Interpreter Module

`modules/emotion_interpreter.py` detects the emotional tone of each transcript
and formats it as ``this is **<emotion>** functionning`` so other systems can
react immediately. Use it standalone:

```python
from modules.emotion_interpreter import interpret_text

result = interpret_text("I feel wonderful about this launch!")
print(result.formatted)  # this is **happy** functionning
```

Or wrap a live speech callback to obtain both transcripts and emotions in real
time:

```python
from modules.emotion_interpreter import emotion_aware_handler
from modules.live_speech import LiveSpeechStreamer

def handle_emotion(result, event):
    print(result.formatted, "| transcript:", event.text)

streamer = LiveSpeechStreamer()
streamer.start(
    emotion_aware_handler(on_emotion=handle_emotion)
)
```

**Run command:**
```bash
python -c "from modules.emotion_interpreter import interpret_text; texts = ['I am so happy today!', 'This is terrible and awful.', 'Wow, that was surprising!']; [print(f'Text: {t}\\nResult: {interpret_text(t).formatted}\\n') for t in texts]"
```

## Voice Identity Module

`modules/voice_identity.py` lets you collect reference samples and later check
whether a new utterance matches someone who has already spoken. It stores MFCC
embeddings (via optional `librosa` + `numpy`) and can capture samples straight
from the microphone (`sounddevice`) or load audio files. Each speaker is stored
as a tiny centroid vector that continuously blends new samples, keeping storage
minimal while maintaining fast, vectorized recognition. Install the optional
deps:

```bash
pip install librosa soundfile sounddevice
```

Sample workflow:

```python
from modules.voice_identity import VoiceIdentifier

identifier = VoiceIdentifier()
identifier.add_sample_from_file("Alice", "samples/alice.wav")
identifier.add_sample_from_file("Bob", "samples/bob.wav")

result = identifier.identify_from_file("incoming.wav")
print(result.describe())

if result.is_known:
    print("Welcome back,", result.speaker)
else:
    print("New speaker detected")

# upgrade an existing profile with another sentence (no extra storage)
identifier.add_sample_from_file("Alice", "samples/alice_meeting.wav")
```

You can persist and reload databases with `save_database()` / `load_database()`,
and `capture_and_sample()` records fresh sentences live so they can be
recognized later in the session. The identifier refreshes its similarity cache
automatically, so lookups remain O(1) per speaker even as you keep refining
voice prints.

**Run command:**
```bash
python -c "from modules.voice_identity import VoiceIdentifier; identifier = VoiceIdentifier(); print('VoiceIdentifier initialized'); print('Known speakers:', list(identifier.known_speakers)); print('Sample rate:', identifier.sample_rate); print('Ready to add voice samples.')"
```

## Realtime Monitor Module

`modules/realtime_monitor.py` ties everything together for an "all signals" live
session: partial transcripts, final transcripts, emotion tags, and speaker
identification. It supports both Vosk (faster) and Whisper (more accurate).

**Using Vosk (default, faster):**

```python
import time
from modules.realtime_monitor import start_insight_session

session = start_insight_session()  # Uses default Vosk model

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    session.stop()
```

**Using Whisper (recommended for accuracy):**

```python
import time
from modules.realtime_monitor import start_insight_session

session = start_insight_session(
    use_whisper=True,
    whisper_kwargs={
        "model_size": "large-v3",  # Best accuracy
        "device": "cpu",  # "cuda" for GPU
        "compute_type": "int8",
    },
)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    session.stop()
```

Each emitted insight contains:

- `text`: live transcript snippet (partial/final).
- `speaker`: the most likely known speaker (if identification is enabled).
- `emotion.formatted`: string like `this is **happy** functionning`.
- `speaker_scores`: raw cosine similarities for auditing/threshold tuning.

You can pass your own `on_insight` callback when calling
`start_insight_session(on_insight=my_handler, ...)` to integrate with logs,
dashboards, or downstream automations.

**Run command:**
```bash
python -c "from modules.realtime_monitor import start_insight_session; import time; print('Starting realtime insight session (requires microphone, press Ctrl+C to stop)...'); session = start_insight_session(on_insight=lambda i: print(f'[{i.emotion.formatted}] {i.text}')); exec('while True: time.sleep(1)')"
```

Note: This requires a microphone. Press Ctrl+C to stop the session.

## Mouse Controller Module

`modules/mouse_controller.py` exposes high-quality cursor automation built on
non-linear Bézier paths so movements feel human rather than robotic. Install the
optional backend:

```bash
pip install pyautogui
```

Key helpers:

- `move_mouse(x, y, profile=None)` – travel to coordinates using smooth easing.
- `drag_mouse(x, y, button="left")` – drag with curved motion.
- `left_click()`, `right_click()`, `middle_click()`, `double_click()`.
- `click_and_drag((x1, y1), (x2, y2))` – combine move and drag in one call.

You can customize movement using `MotionProfile(duration=0.5, curve_strength=0.3)`
for faster/slower timings or sharper arcs. All functions raise
`BackendUnavailable` when `pyautogui` is not installed, so you can guard usage
with `mouse_controller.backend_available()`.

**Run command:**
```bash
python -c "from modules.mouse_controller import move_mouse, MotionProfile; import pyautogui; pos = pyautogui.position(); print(f'Current mouse position: {pos}'); profile = MotionProfile(duration=0.5, curve_strength=0.3); print(f'Moving mouse with profile: duration={profile.duration}s'); move_mouse(pos.x + 100, pos.y + 100, profile=profile); print(f'New mouse position: {pyautogui.position()}')"
```

## Keyboard Controller Module

`modules/keyboard_controller.py` exposes a `Keyboard` class where each key is a
callable attribute. The backend is also `pyautogui`, so install it if you want
to automate typing:

```python
from modules.keyboard_controller import Keyboard

kb = Keyboard()
kb.ctrl(action="down")
kb.c()              # taps 'c' while ctrl is held
kb.ctrl(action="up")

kb.combo("shift", "alt", "tab")  # press simultaneously
kb.space()                       # tap once
kb.pressed_keys()                # ('shift', 'alt', 'tab', ...) if still held
```

You can press multiple keys at the same time via `combo`, hold keys (`action="down"`)
and release them later (`action="up"`), or call `release_all()` to reset the
state. Use `pressed_keys()` to inspect what is currently held and
`keyboard_controller.backend_available()` to check whether `pyautogui` is
available before invoking these helpers.

**Run command:**
```bash
python -c "from modules.keyboard_controller import default_keyboard; kb = default_keyboard(); print('Keyboard initialized'); print('Available methods: space, enter, ctrl, shift, etc.'); print('Example: kb.space() to press space, kb.combo(\"ctrl\", \"c\") for Ctrl+C')"
```

## Text Recognition Module

`modules/text_recognition.py` wraps EasyOCR so you can extract text from clean
documents or warped signage alike. Install the optional deps:

```bash
pip install easyocr opencv-python-headless Pillow
```

Example use:

```python
from modules.text_recognition import extract_text, recognize_text

print(extract_text("photos/menu.jpg"))

for item in recognize_text("photos/menu.jpg"):
    print(item.text, item.confidence, item.box)
```

**Run command:**
```bash
python -c "from modules.text_recognition import extract_text; print(extract_text('path/to/your/image.jpg'))"
```

Replace `'path/to/your/image.jpg'` with the actual path to your image file.

The recognizer automatically loads the EasyOCR model for English by default; pass
`languages=['en', 'fr']` or set `gpu=True` if you have GPU acceleration
available.

## Video Text Recognition Module

`modules/video_text_recognition.py` samples frames from a video, applies the
same enhanced preprocessing pipeline as the still-image OCR, and uses temporal
voting to stabilize results when letters rotate or move. Install the PaddleOCR
dependencies (see the previous section), then:

```python
from modules.video_text_recognition import recognize_video_text

result = recognize_video_text("videos/rotating_logo.mp4", sample_rate=3.0)
print("Dominant:", result.dominant_text)
print("Candidates:", result.candidates)
for frame in result.frames[:5]:
    print(frame.timestamp, frame.text, frame.confidence)
```

**Run command:**
```bash
python -c "from modules.video_text_recognition import recognize_video_text; result = recognize_video_text('path/to/your/video.mp4', sample_rate=3.0); print('Dominant text:', result.dominant_text); print('Top candidates:', result.candidates[:3])"
```

Replace `'path/to/your/video.mp4'` with the actual path to your video file.

Note: Replace `clip.mp4` with a real video file path for actual testing.

The recognizer automatically skips redundant frames, boosts ones with motion,
tries multiple rotations per frame, and votes for the most frequent/highest
confidence string, which makes it resilient against rotating 3D glyphs or
scrolling marquees. Adjust `sample_rate`, `max_frames`, or `motion_threshold`
when processing very long or extremely dynamic videos.

## Morpion Solver Module

`modules/morpion_solver.py` exposes perfect-play tic-tac-toe logic with a simple
API:

```python
from modules.morpion_solver import best_move

board = [
    "XO ",
    " X ",
    "  O",
]
print(best_move(board))        # infers player's turn automatically
print(best_move(board, "O"))   # force evaluation for O
```

It returns `(row, col)` pairs (0-indexed). Games that are already finished raise
helpful errors, and the solver will win/block immediately before falling back to
minimax search. This allows you to plug it into UI automation, reinforcement
learning environments, or debugging tools without reinventing the strategy.

**Run command:**
```bash
python -c "from modules.morpion_solver import best_move, parse_board, board_to_rows, place_move, winner; board = ['XO ', ' X ', '  O']; print('Current board:'); [print(row) for row in board]; move = best_move(board); print(f'Best move for current player: {move}'); new_board = place_move(parse_board(board), move, 'X' if board.count('X') == board.count('O') else 'O'); print('Board after move:'); [print(row) for row in board_to_rows(new_board)]"
```

