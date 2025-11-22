"""
Voice identity sampling and recognition helpers.

The module lets you sample short utterances, convert them into embeddings, and
later check if an incoming clip matches any known speaker. It uses lightweight
MFCC fingerprints via the optional ``librosa`` package and can capture raw audio
through ``sounddevice`` when available.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None  # type: ignore


class BackendUnavailable(RuntimeError):
    """Raised when a required audio backend is missing."""


@dataclass
class VoicePrint:
    """Persistent representation of a known speaker voice."""

    name: str
    embedding: List[float]
    sample_rate: int
    sample_count: int = 1
    metadata: Optional[dict] = None

    def similarity(self, other_embedding: Sequence[float]) -> float:
        """Return cosine similarity against another embedding."""
        _ensure_numpy()
        a = np.asarray(self.embedding, dtype=np.float32)
        b = np.asarray(other_embedding, dtype=np.float32)
        if not a.size or not b.size:
            return 0.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def integrate(self, new_embedding: Sequence[float]) -> None:
        """Merge a new embedding while keeping storage minimal."""
        _ensure_numpy()
        base = np.asarray(self.embedding, dtype=np.float32)
        incoming = np.asarray(new_embedding, dtype=np.float32)
        if not base.size:
            self.embedding = incoming.tolist()
            self.sample_count = 1
            return
        self.sample_count += 1
        alpha = 1.0 / self.sample_count
        blended = base + alpha * (incoming - base)
        norm = np.linalg.norm(blended)
        if norm:
            blended /= norm
        self.embedding = blended.tolist()


@dataclass
class IdentificationResult:
    """Outcome of an identification attempt."""

    speaker: Optional[str]
    confidence: float
    is_known: bool
    scores: Dict[str, float]

    def describe(self) -> str:
        """Return a human-friendly summary."""
        if not self.speaker:
            return "Unknown speaker"
        return f"Speaker '{self.speaker}' confidence={self.confidence:.2f}"


def _ensure_librosa() -> None:
    if librosa is None:
        raise BackendUnavailable(
            "librosa is required for voice fingerprinting. Install it via "
            "`pip install librosa soundfile`."
        )


def _ensure_sounddevice() -> None:
    if sd is None:
        raise BackendUnavailable(
            "sounddevice is required for live audio capture. Install it via "
            "`pip install sounddevice`."
        )


def _ensure_numpy() -> None:
    if np is None:
        raise BackendUnavailable(
            "numpy is required for voice fingerprinting. Install it via "
            "`pip install numpy`."
        )


def _to_numpy(sequence: Sequence[float]) -> np.ndarray:
    """Convert an arbitrary sequence to a numpy vector."""
    _ensure_numpy()
    return np.asarray(sequence, dtype=np.float32)


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    """Return a unit-length numpy vector."""
    _ensure_numpy()
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm:
        arr = arr / norm
    return arr


def _fingerprint(audio: Sequence[float], sample_rate: int) -> List[float]:
    """Compute a normalized MFCC fingerprint for the provided audio."""
    _ensure_numpy()
    _ensure_librosa()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    stats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    norm = np.linalg.norm(stats)
    if norm == 0:
        return stats.tolist()
    return (stats / norm).tolist()


def _load_audio(path: str | Path, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Load an audio file using librosa."""
    _ensure_numpy()
    _ensure_librosa()
    audio, sr = librosa.load(path, sr=sample_rate)
    return audio, sr


class VoiceIdentifier:
    """Manage voice samples and perform recognition."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self._prints: Dict[str, VoicePrint] = {}
        self._embedding_cache: Optional[np.ndarray] = None
        self._embedding_names: List[str] = []

    def _invalidate_cache(self) -> None:
        self._embedding_cache = None
        self._embedding_names = []

    def _rebuild_cache(self) -> None:
        _ensure_numpy()
        names = list(self._prints.keys())
        if not names:
            self._embedding_cache = np.zeros((0, 0), dtype=np.float32)
            self._embedding_names = []
            return
        matrix = np.stack([_normalize_vector(vp.embedding) for vp in self._prints.values()]).astype(np.float32)
        self._embedding_cache = matrix
        self._embedding_names = names

    def _get_embedding_cache(self) -> tuple[np.ndarray, List[str]]:
        if self._embedding_cache is None:
            self._rebuild_cache()
        assert self._embedding_cache is not None
        return self._embedding_cache, self._embedding_names

    # ---------------------------------------------------------------- sampling
    def add_sample(
        self,
        name: str,
        audio: Sequence[float],
        *,
        sample_rate: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> VoicePrint:
        """Register a new voice sample."""
        sr = sample_rate or self.sample_rate
        embedding = _fingerprint(_to_numpy(audio), sr)
        normalized = _normalize_vector(embedding).tolist()
        if name in self._prints:
            self._prints[name].integrate(normalized)
            voice_print = self._prints[name]
        else:
            voice_print = VoicePrint(
                name=name,
                embedding=normalized,
                sample_rate=sr,
                metadata=metadata,
            )
            self._prints[name] = voice_print
        self._invalidate_cache()
        return voice_print

    def add_sample_from_file(
        self,
        name: str,
        path: str | Path,
        *,
        sample_rate: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> VoicePrint:
        """Load audio from disk and register it."""
        sr = sample_rate or self.sample_rate
        audio, _ = _load_audio(path, sr)
        return self.add_sample(name, audio, sample_rate=sr, metadata=metadata)

    def capture_and_sample(
        self,
        name: str,
        *,
        duration: float = 3.0,
        sample_rate: Optional[int] = None,
        device: Optional[int | str] = None,
        metadata: Optional[dict] = None,
    ) -> VoicePrint:
        """Record a short utterance from the microphone and register it."""
        _ensure_sounddevice()
        sr = sample_rate or self.sample_rate
        frames = int(duration * sr)
        recording = sd.rec(frames, samplerate=sr, channels=1, dtype="float32", device=device)
        sd.wait()
        audio = recording[:, 0]
        return self.add_sample(name, audio, sample_rate=sr, metadata=metadata)

    # ----------------------------------------------------------- identification
    def identify(
        self,
        audio: Sequence[float],
        *,
        sample_rate: Optional[int] = None,
        threshold: float = 0.75,
    ) -> IdentificationResult:
        """Return the most likely speaker for the given audio."""
        sr = sample_rate or self.sample_rate
        if not self._prints:
            return IdentificationResult(speaker=None, confidence=0.0, is_known=False, scores={})
        embedding = _normalize_vector(_fingerprint(_to_numpy(audio), sr))
        matrix, names = self._get_embedding_cache()
        if matrix.size == 0:
            return IdentificationResult(speaker=None, confidence=0.0, is_known=False, scores={})
        scores_array = matrix @ embedding
        scores = {name: float(score) for name, score in zip(names, scores_array)}
        speaker, confidence = max(scores.items(), key=lambda item: item[1])
        is_known = confidence >= threshold
        return IdentificationResult(speaker=speaker if is_known else None, confidence=confidence, is_known=is_known, scores=scores)

    def identify_from_file(
        self,
        path: str | Path,
        *,
        sample_rate: Optional[int] = None,
        threshold: float = 0.75,
    ) -> IdentificationResult:
        """Convenience wrapper that loads audio from disk and identifies it."""
        sr = sample_rate or self.sample_rate
        audio, _ = _load_audio(path, sr)
        return self.identify(audio, sample_rate=sr, threshold=threshold)

    # --------------------------------------------------------------- persistence
    def save_database(self, path: str | Path) -> None:
        """Persist the known voice prints to disk."""
        data = {
            "sample_rate": self.sample_rate,
            "voices": [
                {
                    "name": vp.name,
                    "embedding": vp.embedding,
                    "sample_rate": vp.sample_rate,
                    "sample_count": vp.sample_count,
                    "metadata": vp.metadata,
                }
                for vp in self._prints.values()
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load_database(cls, path: str | Path) -> "VoiceIdentifier":
        """Load a previously saved database."""
        payload = json.loads(Path(path).read_text())
        identifier = cls(sample_rate=payload.get("sample_rate", 16_000))
        for item in payload.get("voices", []):
            identifier._prints[item["name"]] = VoicePrint(
                name=item["name"],
                embedding=item["embedding"],
                sample_rate=item["sample_rate"],
                sample_count=item.get("sample_count", 1),
                metadata=item.get("metadata"),
            )
        identifier._invalidate_cache()
        return identifier

    # ---------------------------------------------------------------- utilities
    @property
    def known_speakers(self) -> Iterable[str]:
        return tuple(self._prints.keys())

    def has_sample(self, name: str) -> bool:
        return name in self._prints


def backend_available() -> Dict[str, bool]:
    """Report availability of optional dependencies."""
    return {
        "numpy": np is not None,
        "librosa": librosa is not None,
        "sounddevice": sd is not None,
    }

