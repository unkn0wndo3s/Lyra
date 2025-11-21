"""Real-time input orchestration with asyncio and thread/process pools."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from lyra.input import (
    ExternalSensorConfiguration,
    InputCaptureError,
    InputManager,
    InputResult,
    InputType,
)

logger = logging.getLogger(__name__)


ProducerMode = Literal["thread", "process", "async", "sync"]
Handler = Callable[[InputResult], Awaitable[None] | None]


@dataclass
class InputSourceConfig:
    """Represents a continuously running input producer."""

    name: str
    input_type: InputType
    producer: Callable[[], Any] | Callable[[], Awaitable[Any]]
    interval: float = 0.0
    mode: ProducerMode = "thread"
    max_events: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    events_emitted: int = 0


class RealTimeProcessor:
    """Coordinates multi-modal input capture using asyncio pipelines."""

    def __init__(
        self,
        *,
        input_manager: Optional[InputManager] = None,
        thread_workers: int = 4,
        process_workers: Optional[int] = None,
        queue_maxsize: int = 128,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.input_manager = input_manager or InputManager()
        self.loop = loop or asyncio.get_event_loop()
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_workers)
        self.process_pool = (
            ProcessPoolExecutor(max_workers=process_workers)
            if process_workers
            else None
        )
        self.queue: asyncio.Queue[InputResult] = asyncio.Queue(maxsize=queue_maxsize)
        self._sources: Dict[str, InputSourceConfig] = {}
        self._source_tasks: Dict[str, asyncio.Task] = {}
        self._handlers: Dict[InputType, List[Handler]] = {}
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._running = False

    # --------------------------------------------------------------------- #
    # Source registration helpers
    # --------------------------------------------------------------------- #

    def register_source(self, config: InputSourceConfig) -> None:
        """Register a generic input source."""
        if config.name in self._sources:
            raise ValueError(f"Source '{config.name}' is already registered.")
        self._sources[config.name] = config
        if self._running:
            self._source_tasks[config.name] = asyncio.create_task(
                self._source_loop(config)
            )

    def unregister_source(self, name: str) -> None:
        """Stop and remove a registered source."""
        config = self._sources.pop(name, None)
        task = self._source_tasks.pop(name, None)
        if task:
            task.cancel()
        if config:
            logger.debug("Unregistered source '%s'", name)

    def register_text_source(
        self,
        name: str,
        supplier: Callable[[], str],
        *,
        interval: float = 0.0,
        mode: ProducerMode = "thread",
        max_events: Optional[int] = None,
    ) -> None:
        """Continuously capture scripted text via the InputManager."""

        def producer() -> InputResult:
            text = supplier()
            return self.input_manager.add_text(text=text)

        self.register_source(
            InputSourceConfig(
                name=name,
                input_type=InputType.TEXT,
                producer=producer,
                interval=interval,
                mode=mode,
                max_events=max_events,
            )
        )

    def register_speech_source(
        self,
        name: str,
        *,
        from_microphone: bool = True,
        file_path: Optional[str] = None,
        language: str = "en-US",
        engine: str = "google",
        interval: float = 0.0,
        mode: ProducerMode = "thread",
        max_events: Optional[int] = None,
    ) -> None:
        """Register a speech transcription stream."""

        def producer() -> InputResult:
            return self.input_manager.transcribe_speech(
                from_microphone=from_microphone,
                file_path=file_path,
                language=language,
                engine=engine,
            )

        self.register_source(
            InputSourceConfig(
                name=name,
                input_type=InputType.SPEECH,
                producer=producer,
                interval=interval,
                mode=mode,
                max_events=max_events,
            )
        )

    def register_image_source(
        self,
        name: str,
        supplier: Callable[[], str],
        *,
        color_mode: int = 1,
        interval: float = 0.0,
        preprocess: Optional[Callable[[Any], Any]] = None,
        mode: ProducerMode = "thread",
        max_events: Optional[int] = None,
    ) -> None:
        """Register an image ingestion stream from disk or another path supplier."""

        def producer() -> InputResult:
            path = supplier()
            return self.input_manager.capture_image(
                file_path=path,
                color_mode=color_mode,
                preprocess=preprocess,
            )

        self.register_source(
            InputSourceConfig(
                name=name,
                input_type=InputType.IMAGE,
                producer=producer,
                interval=interval,
                mode=mode,
                max_events=max_events,
            )
        )

    def register_sensor_stream(
        self,
        name: str,
        sensor_name: str,
        *,
        interval: float = 1.0,
        mode: ProducerMode = "thread",
        max_events: Optional[int] = None,
    ) -> None:
        """Register a sensor capture loop."""

        def producer() -> InputResult:
            return self.input_manager.capture_sensor(sensor_name)

        self.register_source(
            InputSourceConfig(
                name=name,
                input_type=InputType.SENSOR,
                producer=producer,
                interval=interval,
                mode=mode,
                max_events=max_events,
            )
        )

    def register_sensor_callback(
        self,
        *,
        name: str,
        capture_fn: Callable[[], Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience helper to add a new hardware sensor to the InputManager."""
        config = ExternalSensorConfiguration(
            name=name,
            capture_fn=capture_fn,
            metadata=metadata,
        )
        self.input_manager.sensor_provider.register_sensor(config)

    # --------------------------------------------------------------------- #
    # Handler registration
    # --------------------------------------------------------------------- #

    def register_handler(self, input_type: InputType, handler: Handler) -> None:
        """Attach a coroutine or sync callback for a specific modality."""
        self._handlers.setdefault(input_type, []).append(handler)

    # --------------------------------------------------------------------- #
    # Lifecycle management
    # --------------------------------------------------------------------- #

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        for config in self._sources.values():
            self._source_tasks[config.name] = asyncio.create_task(
                self._source_loop(config)
            )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for task in self._source_tasks.values():
            task.cancel()
        await asyncio.gather(*self._source_tasks.values(), return_exceptions=True)
        self._source_tasks.clear()
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatcher_task
            self._dispatcher_task = None
        self.thread_pool.shutdown(wait=False)
        if self.process_pool:
            self.process_pool.shutdown(wait=False)

    async def run_for(self, duration: float) -> None:
        """Start the processor, run for ``duration`` seconds, then stop."""
        await self.start()
        await asyncio.sleep(duration)
        await self.stop()

    # --------------------------------------------------------------------- #
    # Internal loops
    # --------------------------------------------------------------------- #

    async def _invoke_producer(self, config: InputSourceConfig) -> InputResult:
        try:
            if config.mode == "async":
                result = await config.producer()  # type: ignore[func-returns-value]
            elif config.mode == "process" and self.process_pool:
                result = await self.loop.run_in_executor(
                    self.process_pool, config.producer
                )
            elif config.mode == "sync":
                result = config.producer()
            else:
                result = await self.loop.run_in_executor(
                    self.thread_pool, config.producer
                )
        except StopIteration:
            raise
        except Exception as exc:
            raise InputCaptureError(
                f"Source '{config.name}' failed to capture input"
            ) from exc

        if not isinstance(result, InputResult):
            raise TypeError(
                f"Source '{config.name}' must return InputResult, got {type(result)}"
            )
        return result

    async def _source_loop(self, config: InputSourceConfig) -> None:
        try:
            while self._running:
                try:
                    result = await self._invoke_producer(config)
                except StopIteration:
                    logger.info("Source '%s' exhausted", config.name)
                    break
                except InputCaptureError as exc:
                    logger.warning("%s", exc)
                    await asyncio.sleep(max(0.1, config.interval))
                    continue

                await self.queue.put(result)
                config.events_emitted += 1
                if config.max_events and config.events_emitted >= config.max_events:
                    logger.info(
                        "Source '%s' reached max events (%s)",
                        config.name,
                        config.max_events,
                    )
                    break

                if config.interval:
                    await asyncio.sleep(config.interval)
        except asyncio.CancelledError:
            logger.debug("Source '%s' cancelled", config.name)

    async def _dispatch_loop(self) -> None:
        try:
            while self._running or not self.queue.empty():
                result = await self.queue.get()
                handlers = self._handlers.get(result.type, [])
                if not handlers:
                    continue
                await asyncio.gather(
                    *[self._execute_handler(handler, result) for handler in handlers],
                    return_exceptions=True,
                )
        except asyncio.CancelledError:
            logger.debug("Dispatcher cancelled")

    async def _execute_handler(self, handler: Handler, result: InputResult) -> None:
        try:
            output = handler(result)
            if asyncio.iscoroutine(output):
                await output
        except Exception:  # noqa: BLE001
            logger.exception("Handler for %s raised", result.type)


__all__ = ["RealTimeProcessor", "InputSourceConfig"]


