"""SpaCy-powered NLP utilities for Lyra."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Dict, Iterable, Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc

try:
    from spacytextblob.spacytextblob import SpacyTextBlob
except ImportError:  # pragma: no cover
    SpacyTextBlob = None

from .models import (
    EntityData,
    IntentPrediction,
    NLPResult,
    SentimentResult,
    TokenData,
)
from .sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


class NLPProcessor:
    """Wrapper around spaCy with sentiment, intent, and entity helpers."""

    DEFAULT_MODEL = "en_core_web_sm"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        enable_sentiment: bool = True,
        intent_keywords: Optional[Dict[str, Iterable[str]]] = None,
    ) -> None:
        self.model_name = model_name
        self.nlp = self._load_model(model_name)
        self.enable_sentiment = enable_sentiment
        self.intent_keywords = intent_keywords or self._default_intent_keywords()
        self.sentiment_analyzer = SentimentAnalyzer() if enable_sentiment else None

        if enable_sentiment and SpacyTextBlob:
            if "spacytextblob" not in self.nlp.pipe_names:
                self.nlp.add_pipe("spacytextblob")
        elif enable_sentiment and not SpacyTextBlob:
            logger.warning(
                "spacytextblob is not installed; sentiment analysis will be disabled."
            )
            self.enable_sentiment = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, text: str) -> NLPResult:
        doc = self.nlp(text)
        tokens = [
            TokenData(
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
            )
            for token in doc
        ]
        entities = [
            EntityData(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
            )
            for ent in doc.ents
        ]
        sentiment = self._resolve_sentiment(doc)
        intent = self._predict_intent(doc)
        return NLPResult(
            text=text,
            tokens=tokens,
            entities=entities,
            sentiment=sentiment,
            intent=intent,
            doc=doc,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model(self, name: str) -> Language:
        try:
            return spacy.load(name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Falling back to a blank English pipeline.",
                name,
            )
            try:
                import_module(name)
            except Exception:
                pass
            return spacy.blank("en")

    def _resolve_sentiment(self, doc: Doc) -> SentimentResult:
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.analyze(doc.text, doc=doc)
        return SentimentResult(polarity=0.0, subjectivity=0.0, label="neutral")

    def _predict_intent(self, doc: Doc) -> IntentPrediction:
        lowered = doc.text.lower()
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in lowered:
                    confidence = min(1.0, 0.5 + 0.05 * len(keyword))
                    return IntentPrediction(
                        label=intent,
                        confidence=confidence,
                        metadata={"matched": keyword},
                    )
        if doc.text.strip().endswith("?"):
            return IntentPrediction(
                label="question",
                confidence=0.55,
                metadata={"reason": "ending-question-mark"},
            )
        if doc[0].lemma_.lower() in {"set", "run", "execute", "start"}:
            return IntentPrediction(
                label="command",
                confidence=0.6,
                metadata={"reason": "verb-first"},
            )
        return IntentPrediction(label="statement", confidence=0.4, metadata={})

    @staticmethod
    def _default_intent_keywords() -> Dict[str, Iterable[str]]:
        return {
            "greeting": {"hello", "hi", "hey", "good morning", "good evening"},
            "status_request": {"status", "progress", "how is", "update"},
            "shutdown": {"stop", "shutdown", "cancel", "abort"},
            "affirmation": {"yes", "confirm", "sure"},
            "negation": {"no", "don't", "do not", "nah"},
        }


__all__ = ["NLPProcessor"]


