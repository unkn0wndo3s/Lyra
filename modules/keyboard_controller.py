"""
Keyboard automation with per-key callables.

Depends on the optional ``pyautogui`` package. Install via:

    pip install pyautogui

Usage:

    kb = Keyboard()
    kb.space()                  # tap space once
    kb.space(action="down")     # hold space
    kb.space(action="up")       # release space
    kb.combo("ctrl", "shift", "esc")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    import pyautogui  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None  # type: ignore


class BackendUnavailable(RuntimeError):
    """Raised when ``pyautogui`` is not available."""


def backend_available() -> bool:
    return pyautogui is not None


def _ensure_backend() -> None:
    if pyautogui is None:
        raise BackendUnavailable(
            "Keyboard control requires the 'pyautogui' package. Install it via `pip install pyautogui`."
        )


def _normalize_key(name: str) -> str:
    replacements = {
        "plus": "+",
        "minus": "-",
        "underscore": "_",
        "comma": ",",
        "period": ".",
        "dot": ".",
        "semicolon": ";",
        "quote": "'",
        "doublequote": '"',
        "slash": "/",
        "backslash": "\\",
        "space": "space",
        "enter": "enter",
        "return": "enter",
        "tab": "tab",
        "esc": "esc",
        "escape": "esc",
        "backspace": "backspace",
    }
    lowered = name.lower()
    return replacements.get(lowered, lowered.replace("_", " "))


@dataclass
class Keyboard:
    """Expose each key as a callable function via attribute access."""

    _pressed: Dict[str, int] = field(default_factory=dict)

    # ----------------------------------------------------------- low level ops
    def press_and_release(self, key: str, interval: float = 0.0) -> None:
        """Press and release the key exactly once."""
        _ensure_backend()
        pyautogui.press(key, interval=interval)

    def press(self, key: str) -> None:
        """Press (hold) the key until ``release`` is called."""
        _ensure_backend()
        pyautogui.keyDown(key)
        self._pressed[key] = self._pressed.get(key, 0) + 1

    def release(self, key: str) -> None:
        """Release a previously held key."""
        count = self._pressed.get(key, 0)
        if count <= 0:
            return
        _ensure_backend()
        pyautogui.keyUp(key)
        if count == 1:
            self._pressed.pop(key, None)
        else:
            self._pressed[key] = count - 1

    def pressed_keys(self) -> tuple[str, ...]:
        """Return the keys currently held down."""
        return tuple(self._pressed.keys())

    def combo(self, *keys: str) -> None:
        """Press multiple keys at the same time (e.g., ctrl+shift+esc)."""
        if not keys:
            return
        normalized = [_normalize_key(k) for k in keys]
        _ensure_backend()
        for key in normalized:
            pyautogui.keyDown(key)
        for key in reversed(normalized):
            pyautogui.keyUp(key)

    def release_all(self) -> None:
        """Release every key currently held down."""
        for key in list(self._pressed.keys()):
            self.release(key)

    # ---------------------------------------------------------- dynamic keys
    def _make_key_handler(self, key_name: str) -> Callable[..., None]:
        normalized = _normalize_key(key_name)

        def handler(action: str = "tap", interval: float = 0.0) -> None:
            """
            Control a single key.

            Args:
                action: "tap" (default), "press", or "release".
                interval: optional delay passed to pyautogui when tapping.
            """

            act = action.lower()
            if act in {"tap", "press_and_release"}:
                self.press_and_release(normalized, interval=interval)
            elif act in {"press", "down"}:
                self.press(normalized)
            elif act in {"release", "up"}:
                self.release(normalized)
            else:
                raise ValueError("action must be 'tap', 'press', or 'release'")

        handler.__name__ = normalized
        return handler

    def __getattr__(self, item: str) -> Callable[..., None]:
        if item.startswith("__"):
            raise AttributeError(item)
        return self._make_key_handler(item)


def default_keyboard() -> Keyboard:
    """Convenience factory."""
    return Keyboard()

