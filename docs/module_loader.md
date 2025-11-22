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
    └── live_speech.py      # Live speech-to-text streamer (Vosk + sounddevice)
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

# create an instance without worrying about manual imports
streamer = stream(model_path="/path/to/vosk-model")
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

If you change existing modules, call `reload_module("live_speech")` or
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

