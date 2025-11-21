"""Sentiment helpers combining spaCy/TextBlob outputs with VADER."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from textblob import TextBlob

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    nltk = None  # type: ignore
    SentimentIntensityAnalyzer = None  # type: ignore

from .models import SentimentResult, ToneDirective

if TYPE_CHECKING:  # pragma: no cover
    from spacy.tokens import Doc

logger = logging.getLogger(__name__)


def _ensure_vader() -> Optional["SentimentIntensityAnalyzer"]:
    if not SentimentIntensityAnalyzer:
        return None
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        if nltk is None:
            return None
        try:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize VADER: %s", exc)
            return None


@dataclass
class SentimentAnalyzer:
    """Runs multi-source sentiment scoring and tone planning."""

    use_vader: bool = True

    def __post_init__(self) -> None:
        self.vader = _ensure_vader() if self.use_vader else None

    def analyze(self, text: str, *, doc: Optional["Doc"] = None) -> SentimentResult:  # type: ignore[name-defined]
        sources: Dict[str, float] = {}

        polarity, subjectivity = self._spacy_or_textblob(doc, text, sources)
        polarity = float(polarity)
        subjectivity = float(subjectivity)

        if self.vader:
            vader_score = float(self.vader.polarity_scores(text)["compound"])
            sources["vader_compound"] = vader_score
            polarity = (polarity + vader_score) / 2

        label = self._label_from_polarity(polarity)
        tone = self._plan_tone(label, subjectivity)
        return SentimentResult(
            polarity=polarity,
            subjectivity=subjectivity,
            label=label,
            tone=tone,
            sources=sources,
        )

    def _spacy_or_textblob(
        self,
        doc: Optional["Doc"],  # type: ignore[name-defined]
        text: str,
        sources: Dict[str, float],
    ) -> tuple[float, float]:
        if doc is not None and hasattr(doc._, "polarity"):
            sources["spacytextblob"] = float(doc._.polarity)
            return float(doc._.polarity), float(doc._.subjectivity)
        blob = TextBlob(text)
        sources["textblob"] = float(blob.sentiment.polarity)
        return float(blob.sentiment.polarity), float(blob.sentiment.subjectivity)

    @staticmethod
    def _label_from_polarity(polarity: float) -> str:
        if polarity > 0.15:
            return "positive"
        if polarity < -0.15:
            return "negative"
        return "neutral"

    @staticmethod
    def _plan_tone(label: str, subjectivity: float) -> ToneDirective:
        if label == "positive":
            style = "encouraging"
            voice = "warm"
            notes = "mirror positive affect and reinforce progress"
        elif label == "negative":
            if subjectivity > 0.5:
                style = "empathetic"
                voice = "calm"
                notes = "acknowledge feelings and offer assistance"
            else:
                style = "solution-focused"
                voice = "steady"
                notes = "address issues directly with actionable steps"
        else:
            style = "neutral"
            voice = "informative"
            notes = "maintain clarity and invite further input"
        escalate = label == "negative" and subjectivity > 0.6
        return ToneDirective(style=style, voice=voice, escalate=escalate, notes=notes)


__all__ = ["SentimentAnalyzer"]


