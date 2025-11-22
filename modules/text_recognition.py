"""
AI-powered OCR utilities built on EasyOCR.

This module can read plain or curved / stylized text from images using deep
learning models. Install the optional dependencies before use:

    pip install easyocr opencv-python-headless Pillow
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore


class BackendUnavailable(RuntimeError):
    """Raised when EasyOCR is not installed."""


def backend_available() -> bool:
    return easyocr is not None


def _ensure_backend() -> None:
    if easyocr is None:
        raise BackendUnavailable(
            "Text recognition requires `easyocr`. Install it via "
            "`pip install easyocr opencv-python-headless Pillow`."
        )


BoundingBox = List[Tuple[int, int]]


@dataclass
class OCRResult:
    text: str
    confidence: float
    box: BoundingBox


class TextRecognizer:
    """High-level wrapper around EasyOCR."""

    def __init__(
        self,
        languages: Sequence[str] | None = None,
        *,
        gpu: bool = False,
    ) -> None:
        self.languages = list(languages or ["en"])
        self.gpu = gpu
        self._reader: easyocr.Reader | None = None  # type: ignore

    def _get_reader(self) -> "easyocr.Reader":
        if self._reader is None:
            _ensure_backend()
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def recognize(
        self,
        image_path: str | Path,
        *,
        paragraph: bool = False,
        detail: bool = True,
    ) -> List[OCRResult] | List[str]:
        """
        Recognize text from an image.

        Args:
            image_path: path to the source image file.
            paragraph: when True, returns concatenated text paragraphs.
            detail: when False, only text strings are returned.
        """

        reader = self._get_reader()
        results = reader.readtext(str(image_path), detail=detail, paragraph=paragraph)
        if not detail:
            return [str(item) for item in results]

        parsed: List[OCRResult] = []
        for box, text, confidence in results:
            parsed.append(
                OCRResult(
                    text=text.strip(),
                    confidence=float(confidence),
                    box=[tuple(map(int, point)) for point in box],
                )
            )
        return parsed

    def extract_text(
        self,
        image_path: str | Path,
        *,
        separator: str = "\n",
    ) -> str:
        """Return only the detected text, joined by ``separator``."""
        reader = self._get_reader()
        lines = reader.readtext(str(image_path), detail=False, paragraph=False)
        filtered = [line.strip() for line in lines if line and line.strip()]
        return separator.join(filtered)


def recognize_text(
    image_path: str | Path,
    *,
    languages: Sequence[str] | None = None,
    gpu: bool = False,
) -> List[OCRResult]:
    recognizer = TextRecognizer(languages=languages, gpu=gpu)
    results = recognizer.recognize(image_path, detail=True)
    return results  # type: ignore[return-value]


def extract_text(
    image_path: str | Path,
    *,
    languages: Sequence[str] | None = None,
    gpu: bool = False,
    separator: str = "\n",
) -> str:
    recognizer = TextRecognizer(languages=languages, gpu=gpu)
    return recognizer.extract_text(image_path, separator=separator)

