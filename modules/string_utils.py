"""String helpers for exercising the module loader."""

class Greeter:
    """Form greeting messages."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}!"


def shout(message: str) -> str:
    """Return the message in uppercase with emphasis."""
    return message.upper() + "!"

