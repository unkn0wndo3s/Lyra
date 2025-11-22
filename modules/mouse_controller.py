"""
Mouse automation helpers with smooth, non-linear motion curves.

All functions rely on the optional ``pyautogui`` package. Install it with:

    pip install pyautogui

The module exposes high-level helpers to move the mouse along cubic Bézier
curves, perform clicks (left/right/middle), and execute drag gestures without
blocking other components.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import pyautogui  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None  # type: ignore


class BackendUnavailable(RuntimeError):
    """Raised when ``pyautogui`` is not installed."""


Point = Tuple[float, float]
EaseFn = Callable[[float], float]


def backend_available() -> bool:
    """Return True when the optional backend is installed."""
    return pyautogui is not None


def _ensure_backend() -> None:
    if pyautogui is None:
        raise BackendUnavailable(
            "Mouse control requires the 'pyautogui' package. Install it via `pip install pyautogui`."
        )


def _bezier_point(t: float, p0: Point, p1: Point, p2: Point, p3: Point) -> Point:
    """Compute a point along a cubic Bézier curve."""
    u = 1.0 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t
    x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
    y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
    return x, y


def _generate_curve(start: Point, end: Point, steps: int, curve_strength: float) -> List[Point]:
    """Return a Bézier path between start and end with a slight arc."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return [start] * steps

    # Perpendicular vector used to create an arc
    perp_x, perp_y = (-dy, dx)
    perp_len = math.hypot(perp_x, perp_y) or 1.0
    perp_x /= perp_len
    perp_y /= perp_len

    offset = curve_strength * distance
    jitter = distance * 0.02
    c1 = (
        start[0] + dx * 0.25 + perp_x * offset + random.uniform(-jitter, jitter),
        start[1] + dy * 0.25 + perp_y * offset + random.uniform(-jitter, jitter),
    )
    c2 = (
        start[0] + dx * 0.75 - perp_x * offset + random.uniform(-jitter, jitter),
        start[1] + dy * 0.75 - perp_y * offset + random.uniform(-jitter, jitter),
    )

    return [_bezier_point(i / (steps - 1), start, c1, c2, end) for i in range(steps)]


def _ease_in_out(t: float) -> float:
    return 0.5 * (1 - math.cos(math.pi * t))


def _ease_out(t: float) -> float:
    return 1 - (1 - t) * (1 - t)


def _ease_in(t: float) -> float:
    return t * t


EASE_LOOKUP: dict[str, EaseFn] = {
    "linear": lambda t: t,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
}


@dataclass
class MotionProfile:
    """Parameters that control how the cursor travels."""

    duration: float | tuple[float, float] = (0.25, 0.55)
    steps: int = 50
    curve_strength: float | tuple[float, float] = (0.15, 0.35)
    easing: str | tuple[str, ...] = ("ease_in_out", "ease_out", "ease_in")


def clamp_duration(value: float | tuple[float, float]) -> float:
    """Return a randomized duration clamped to <1s for human-like motion."""
    if isinstance(value, tuple):
        low, high = value if len(value) == 2 else (value[0], value[-1])
        low = max(0.08, min(low, 0.95))
        high = max(low, min(high, 0.98))
        return random.uniform(low, high)
    return max(0.08, min(float(value), 0.98))


def pick_curve_strength(value: float | tuple[float, float]) -> float:
    if isinstance(value, tuple):
        low, high = value if len(value) == 2 else (value[0], value[-1])
        return random.uniform(max(0.05, low), min(0.6, high))
    return max(0.05, min(float(value), 0.6))


def pick_easing(value: str | tuple[str, ...]) -> EaseFn:
    if isinstance(value, tuple):
        choice = random.choice(value)
        return EASE_LOOKUP.get(choice, _ease_in_out)
    return EASE_LOOKUP.get(value, _ease_in_out)


def _sleep_until(target_time: float) -> None:
    remaining = target_time - time.perf_counter()
    if remaining > 0:
        time.sleep(min(remaining, 0.01))


def _apply_motion(path: Sequence[Point], duration: float, easing: EaseFn) -> None:
    _ensure_backend()
    if not path:
        return
    start_time = time.perf_counter()
    total_points = len(path)
    for idx, point in enumerate(path, start=1):
        pyautogui.moveTo(point[0], point[1])
        if duration <= 0:
            continue
        progress = easing(idx / total_points)
        target = start_time + progress * duration
        _sleep_until(target)


def move_mouse(x: float, y: float, profile: MotionProfile | None = None) -> None:
    """
    Move the cursor to the given coordinates using a non-linear Bézier curve.

    Args:
        x, y: Screen coordinates.
        profile: MotionProfile controlling duration, curvature, and easing.
    """
    _ensure_backend()
    profile = profile or MotionProfile()
    duration = clamp_duration(profile.duration)
    curve_strength = pick_curve_strength(profile.curve_strength)
    easing = pick_easing(profile.easing)
    start_pos = pyautogui.position()
    path = _generate_curve(start_pos, (x, y), max(2, profile.steps), curve_strength)
    _apply_motion(path, duration, easing)


def drag_mouse(
    x: float,
    y: float,
    *,
    button: str = "left",
    profile: MotionProfile | None = None,
) -> None:
    """Drag the cursor to ``(x, y)`` while holding the specified mouse button."""
    _ensure_backend()
    profile = profile or MotionProfile()
    duration = clamp_duration(profile.duration)
    curve_strength = pick_curve_strength(profile.curve_strength)
    easing = pick_easing(profile.easing)
    start_pos = pyautogui.position()
    path = _generate_curve(start_pos, (x, y), max(2, profile.steps), curve_strength)
    pyautogui.mouseDown(button=button)
    try:
        _apply_motion(path, duration, easing)
    finally:
        pyautogui.mouseUp(button=button)


def click(button: str = "left", clicks: int = 1, interval: float = 0.05) -> None:
    """Perform direct mouse clicks."""
    _ensure_backend()
    pyautogui.click(button=button, clicks=clicks, interval=interval)


def double_click(button: str = "left") -> None:
    """Perform a double-click."""
    click(button=button, clicks=2, interval=0.1)


def right_click() -> None:
    click(button="right")


def left_click() -> None:
    click(button="left")


def middle_click() -> None:
    click(button="middle")


def click_and_drag(
    start: Point,
    end: Point,
    *,
    button: str = "left",
    profile: MotionProfile | None = None,
) -> None:
    """Move the cursor to ``start`` and drag to ``end``."""
    _ensure_backend()
    move_mouse(*start, profile=profile)
    drag_mouse(*end, button=button, profile=profile)

