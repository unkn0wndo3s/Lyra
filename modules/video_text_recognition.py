"""
High-accuracy text recognition for video streams.

This module combines frame sampling, motion-aware preprocessing, PaddleOCR for
robust recognition, and temporal voting to extract text from challenging videos
with rotating or moving characters.

Dependencies (in addition to ``paddleocr`` requirements):

    pip install paddleocr paddlepaddle pillow opencv-python-headless numpy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from modules.text_recognition import _decode_view, _enhance_image, _generate_views, PaddleOCR, backend_available as ocr_backend_available


class BackendUnavailable(RuntimeError):
    """Raised when the required OCR backend is missing."""


def backend_available() -> bool:
    return ocr_backend_available()


def _ensure_backend() -> None:
    if not backend_available():
        raise BackendUnavailable(
            "Video text recognition requires the PaddleOCR dependencies. "
            "Install them via `pip install paddleocr paddlepaddle pillow opencv-python-headless`."
        )


@dataclass
class FrameOCR:
    timestamp: float
    text: str
    confidence: float


@dataclass
class VideoOCRResult:
    dominant_text: str
    candidates: Dict[str, int]
    frames: List[FrameOCR] = field(default_factory=list)


class VideoTextRecognizer:
    """Extracts text from videos via frame sampling and temporal voting."""

    def __init__(
        self,
        *,
        sample_rate: float = 2.0,
        max_frames: int = 200,
        motion_threshold: float = 5.0,
        languages: Sequence[str] | None = None,
    ) -> None:
        self.sample_rate = max(0.5, sample_rate)
        self.max_frames = max_frames
        self.motion_threshold = motion_threshold
        lang = languages[0] if languages else "en"
        self.lang = lang
        self._ocr: PaddleOCR | None = None  # type: ignore

    def _get_ocr(self) -> PaddleOCR:
        if self._ocr is None:
            _ensure_backend()
            self._ocr = PaddleOCR(lang=self.lang, ocr_version="PP-OCRv4")
        return self._ocr

    def recognize(self, video_path: str | Path) -> VideoOCRResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps / self.sample_rate))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        ocr = self._get_ocr()
        prev_gray: Optional[np.ndarray] = None
        frames: List[FrameOCR] = []
        candidate_counts: Dict[str, int] = {}

        current_index = 0
        captured = 0

        while cap.isOpened() and captured < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if current_index % frame_interval != 0:
                current_index += 1
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if self._should_process(frame, prev_gray):
                text, confidence = self._process_frame(frame, ocr)
                if text:
                    frames.append(FrameOCR(timestamp=timestamp, text=text, confidence=confidence))
                    candidate_counts[text] = candidate_counts.get(text, 0) + 1
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            captured += 1
            current_index += 1

        cap.release()

        dominant_text = self._select_dominant(candidate_counts, frames)
        return VideoOCRResult(dominant_text=dominant_text, candidates=candidate_counts, frames=frames)

    def _should_process(self, frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> bool:
        if prev_gray is None:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        score = diff.mean()
        return score >= self.motion_threshold

    def _process_frame(self, frame: np.ndarray, ocr: PaddleOCR) -> Tuple[str, float]:
        enhanced = _enhance_image(frame)
        color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        best_text = ""
        best_score = -math.inf
        for view in [color, cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(color, cv2.ROTATE_180), cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
            text, score = _decode_view(ocr, view)
            if score > best_score:
                best_text = text
                best_score = score
        return best_text, best_score

    def _select_dominant(self, counts: Dict[str, int], frames: List[FrameOCR]) -> str:
        if not counts:
            return ""
        sorted_candidates = sorted(counts.items(), key=lambda item: (-item[1], -self._average_conf(item[0], frames)))
        return sorted_candidates[0][0]

    @staticmethod
    def _average_conf(text: str, frames: List[FrameOCR]) -> float:
        rel = [frame.confidence for frame in frames if frame.text == text]
        if not rel:
            return 0.0
        return sum(rel) / len(rel)


def recognize_video_text(
    video_path: str | Path,
    *,
    sample_rate: float = 2.0,
    max_frames: int = 200,
) -> VideoOCRResult:
    recognizer = VideoTextRecognizer(sample_rate=sample_rate, max_frames=max_frames)
    return recognizer.recognize(video_path)

