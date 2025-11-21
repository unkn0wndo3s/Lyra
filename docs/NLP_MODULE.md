# NLP & Dialogue Module

Lyra's NLP stack is powered by spaCy for tokenisation, parsing, and named
entity recognition, with optional sentiment analysis supplied by
`spacytextblob`. Higher-level context is managed via the `DialogueManager`,
which links the NLP pipeline with the memory subsystem.

## Components

| Component | Description |
|-----------|-------------|
| `NLPProcessor` | Wraps a spaCy model (`en_core_web_sm` by default) to generate structured tokens, entities, sentiment, and intent predictions. |
| `DialogueManager` | Maintains session context, persists short-term state (conversation turns) and optionally archives turns to long-term memory. |
| `NLPResult` | Dataclass describing the outcome (tokens, entities, sentiment, intent). |
| `DialogueContext` | Rolling window of conversation turns; stored in short-term memory with a configurable TTL. |

## Usage

```python
from lyra import DialogueManager

dialogue = DialogueManager()
response = dialogue.process_input("Hello Lyra, how's the status?")

print(response.nlp.intent.label)        # e.g. "greeting"
print(response.nlp.sentiment.label)     # e.g. "positive"
print(len(response.context.turns))      # context window
print(response.nlp.sentiment.tone.style)  # e.g. "empathetic"
```

### Sentiment & Intent

- Sentiment uses `spacytextblob` when available. Install via
  `./venv/bin/pip install spacytextblob` (already bundled in this project).
- Intent detection currently relies on keyword heuristics plus structural cues
  (question marks, imperative verbs). Extend `intent_keywords` when
  instantiating `NLPProcessor` for domain-specific intents.
- Additional tone guidance is produced by the `SentimentAnalyzer`, which fuses
  spaCy/TextBlob scores with optional VADER estimates. The resulting
  `SentimentResult.tone` recommends response style/voice and flags when to
  escalate (e.g., highly negative, subjective statements).

### Memory Integration

- Short-term context is stored under keys like `dialogue:<session_id>` with a
  default TTL of 15 minutes. This enables other subsystems (e.g., task planners)
  to fetch the latest dialogue state.
- Long-term storage writes each turn into SQLite (`category='dialogue'`)
  including entities and sentiment metadata. Entries expire after 30 days by
  default and are subject to the existing memory garbage collector.

### Model Download

Ensure the spaCy model is available:

```bash
./venv/bin/python -m spacy download en_core_web_sm
```

If the model is missing, Lyra falls back to a blank English pipeline (tokeniser
only), which still works but without POS tags/NER accuracy.


