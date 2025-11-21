# Adaptive Learning Module

Lyra can capture user feedback after every interaction and fold it back into
future responses. The adaptive learning layer currently tracks rewards per
intent, infers tone/voice preferences, and stores all evidence in long-term
memory for later training.

## Components

| Symbol | Description |
|--------|-------------|
| `FeedbackRecord` | Captures a single reward signal for a response (positive/negative numeric reward). |
| `PolicyStats` | Aggregated statistics per intent (counts, average reward, tone preference). |
| `AdaptiveLearner` | Facade for recording feedback and requesting updated policy guidance. |

## Recording Feedback

```python
from lyra import AdaptiveLearner, FeedbackRecord

learner = AdaptiveLearner()
record = FeedbackRecord(
    session_id="demo",
    interaction_id="demo-001",
    input_text="Give me the status",
    response_text="Everything is green",
    reward=0.8,
    intent="status_request",
    sentiment="positive",
    metadata={"tone": "encouraging", "voice": "warm"},
)
learner.record_feedback(record)
```

Each feedback item is appended to `learning:feedback:<intent>` in long-term
memory and truncated to the most recent N samples (default 200). Aggregated
stats live at `learning:stats:<intent>` with evolving averages and preferences.

## Using Recommendations

```python
recommendation = learner.suggest_policy("status_request")
print(recommendation.tone, recommendation.voice, recommendation.confidence)
```

Integrate this with the response generator by passing the recommended tone or
voice into `ResponsePlan` (e.g., extend `ResponseGenerator` to query
`AdaptiveLearner` before crafting the style guide).

## Future Extensions

- Replace the rule-based stats with a true reinforcement learner (e.g.,
  contextual bandits) using the stored reward history.
- Train supervised fine-tuning datasets from the persisted `FeedbackRecord`
  rows (they already contain user input, system response, and reward).
- Use memory tags to bucket feedback by sentiment, persona, or task type, then
  compute specialized policies per bucket.


