"""Collection of math helpers used by module_loader.py tests."""

class Vector2D:
    """Simple 2D vector for demonstration purposes."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b

