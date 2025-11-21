# Real-Time Processing

Lyra's runtime pipeline uses `asyncio` plus thread/process pools to capture
multi-modal input without blocking the main agent loop.

## Core Class: `RealTimeProcessor`

Key capabilities:

- **Concurrent sources:** register independent producers for text, speech,
  images, or sensors. Each runs inside its own asynchronous task.
- **Thread/process pools:** blocking providers (e.g., microphone, OpenCV) run
  in background workers, while CPU-heavy workloads can opt into a process pool.
- **Async handlers:** register coroutine or synchronous callbacks per
  `InputType`. Dispatch happens via an internal queue to ensure back-pressure.
- **Lifecycle helpers:** `start()`, `stop()`, and `run_for(seconds)` manage the
  underlying tasks and executors.

## Typical Usage

```python
import asyncio
from lyra import InputManager, InputType, RealTimeProcessor

async def main():
    inputs = InputManager()
    processor = RealTimeProcessor(input_manager=inputs)

    # Register handlers
    async def handle_text(result):
        print("text:", result.content)

    processor.register_handler(InputType.TEXT, handle_text)

    # Register text source that streams scripted commands
    commands = iter(["status report", "init diagnostics"])
    processor.register_text_source("scripted-text", supplier=lambda: next(commands))

    await processor.run_for(0.5)

asyncio.run(main())
```

## Custom Sources & Sensors

- `register_source(InputSourceConfig(...))` accepts custom producers, including
  async coroutines. Use `mode="process"` for CPU-heavy work.
- `register_sensor_callback` wires external hardware directly into the shared
  `ExternalSensorProvider`, making the data available to both the input module
  and the real-time processor.

## Error Handling & Back-Pressure

- Producers that raise `StopIteration` automatically stop their stream.
- Any exception inside a producer is surfaced as `InputCaptureError` and logged;
  the processor keeps running unless the error repeats.
- The internal queue size (default 128) prevents unbounded memory usage. If a
  handler is slow, producers will naturally pause until space is available.


