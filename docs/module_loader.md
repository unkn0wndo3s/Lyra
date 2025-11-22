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
    └── keyboard_controller.py # Human-like keyboard automation
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

streamer = stream(model_path="/path/to/vosk-model")
emotion = emotion_interpret("I am thrilled to be here!")
print(emotion.formatted)  # this is **happy** functionning
identifier = voice_identifier_cls()
print(identifier.known_speakers)  # ()
# launch full realtime session (requires optional deps)
session = insight_session()
mouse_move(200, 300)
keyboard().space()
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
and the `sounddevice` library; install both and download a Vosk model before
starting a stream:

```bash
pip install vosk sounddevice
export VOSK_MODEL_PATH=/path/to/vosk-model-small-en-us-0.15
```

Then start listening:

```python
from modules.live_speech import LiveSpeechStreamer

def on_transcript(event):
    print("FINAL" if event.is_final else "LIVE", event.text)

streamer = LiveSpeechStreamer()
streamer.start(on_transcript)
```

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

## Realtime Monitor Module

`modules/realtime_monitor.py` ties everything together for an "all signals" live
session: partial transcripts, final transcripts, emotion tags, and speaker
identification. It depends on whatever backends each sub-system requires
(install the optional packages mentioned earlier).

```python
import time
from modules.realtime_monitor import start_insight_session
from modules.voice_identity import VoiceIdentifier

identifier = VoiceIdentifier()
identifier.add_sample_from_file("Alice", "samples/alice.wav")

session = start_insight_session(
    voice_identifier=identifier,
    stream_kwargs={"model_path": "/path/to/vosk-model-small-en-us-0.15"},
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

