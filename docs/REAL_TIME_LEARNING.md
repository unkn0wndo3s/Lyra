# Real-Time Learning

The real-time learning engine monitors dialogue streams and records new facts
into memory without blocking the main interaction loop. It uses lightweight,
rule-based extractors that run synchronously within milliseconds.

## RealTimeLearner

```python
from lyra import DialogueManager, RealTimeLearner

dialogue = DialogueManager()
learner = RealTimeLearner(memory_manager=dialogue.memory_manager)

response = dialogue.process_input("My name is Alex and the deployment status is green.", session_id="demo")
facts = learner.learn_from_dialogue(response, session_id="demo")
print([fact.key for fact in facts])
```

### Extraction Rules

- **Identity**: captures phrases like “my name is X” or “call me Y” and stores
  them under `user:profile:name`.
- **Status statements**: phrases containing “status … is …” produce session-
  scoped facts with a short TTL.
- **Named entities**: PERSON/ORG/GPE entities are stored with the utterance
  context for future reference.

### Storage Strategy

- Short-term memory receives the fact with a TTL (for quick access).
- Long-term memory archives the same fact with tags and categories so it can be
  retrieved later via `MemoryRetriever`.
- Importance metadata is preserved so prioritization logic stays consistent.

### Extensibility

Add more rules by subclassing `RealTimeLearner` or by extending the `_extract_*`
helpers. Because the learner uses the existing `MemoryManager`, updates happen
within the same transaction system as the rest of Lyra, ensuring consistency
and minimal latency.


