"""
AI-powered OCR utilities built on PaddleOCR.

This module can read plain or curved / stylized text from images using deep
learning models. Install the optional dependencies before use:

    pip install paddleocr paddlepaddle pillow opencv-python-headless
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore


class BackendUnavailable(RuntimeError):
    """Raised when EasyOCR is not installed."""


def backend_available() -> bool:
    return PaddleOCR is not None


def _ensure_backend() -> None:
    if PaddleOCR is None:
        raise BackendUnavailable(
            "Text recognition now relies on PaddleOCR. Install it via "
            "`pip install paddleocr paddlepaddle pillow opencv-python-headless`."
        )


BoundingBox = List[Tuple[int, int]]


@dataclass
class OCRResult:
    text: str
    confidence: float
    box: BoundingBox


class TextRecognizer:
    """High-level wrapper around PaddleOCR."""

    def __init__(
        self,
        languages: Sequence[str] | None = None,
        *,
        gpu: bool = False,
    ) -> None:
        lang = languages[0] if languages else "en"
        self.lang = lang
        self.gpu = gpu
        self._reader: PaddleOCR | None = None  # type: ignore

    def _get_reader(self) -> "PaddleOCR":
        if self._reader is None:
            _ensure_backend()
            self._reader = PaddleOCR(
                lang=self.lang,
                ocr_version="PP-OCRv4",
            )
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
        views = _generate_views(image_path)

        best_result: OCRResult | None = None
        best_score = -1.0

        for view in views:
            candidate_text, candidate_score = _decode_view(reader, view)
            if not candidate_text:
                continue
            if candidate_score > best_score:
                h, w = view.shape[:2]
                box = [(0, 0), (w, 0), (w, h), (0, h)]
                best_result = OCRResult(text=candidate_text, confidence=candidate_score, box=box)
                best_score = candidate_score

        if best_result is None:
            return [] if detail else []

        if not detail:
            return [best_result.text]

        if paragraph:
            return [best_result]

        return [best_result]

    def extract_text(
        self,
        image_path: str | Path,
        *,
        separator: str = "\n",
    ) -> str:
        """Return only the detected text, joined by ``separator``."""
        texts = self.recognize(image_path, detail=False)
        filtered = [line.strip() for line in texts if line and line.strip()]
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


def _generate_views(image_path: str | Path) -> List[np.ndarray]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")

    enhanced = _enhance_image(img)
    color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    rotations = [
        color,
        cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(color, cv2.ROTATE_180),
        cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    return rotations


def _enhance_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    h, w = inverted.shape[:2]
    scale = max(2.5, min(4.5, 900.0 / max(h, w)))
    resized = cv2.resize(inverted, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        -12,
    )
    dilated = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
    median = cv2.medianBlur(dilated, 3)
    inverted_again = cv2.bitwise_not(median)
    final = cv2.adaptiveThreshold(
        inverted_again,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        33,
        5,
    )
    return final


def _decode_view(reader: "PaddleOCR", view: np.ndarray) -> Tuple[str, float]:
    raw = reader.ocr(view)
    texts: List[str] = []
    confidences: List[float] = []

    for entry in raw or []:
        entry_texts = entry.get("rec_texts") or []
        entry_scores = entry.get("rec_scores") or []
        for idx, text in enumerate(entry_texts):
            clean = text.strip()
            if not clean:
                continue
            texts.append(clean)
            confidences.append(float(entry_scores[idx]) if idx < len(entry_scores) else 0.0)

    candidate = "".join(texts)
    if not candidate:
        return "", 0.0

    alnum_ratio = sum(ch.isalnum() for ch in candidate) / max(1, len(candidate))
    avg_conf = sum(confidences) / max(1, len(confidences))
    score = len(candidate) * (avg_conf + alnum_ratio)
    return candidate, score

