# Response Generation Module

The response module produces conversational replies that adapt to user intent,
sentiment, and historical context. It combines rule-based fallbacks with
optional large-language-model (LLM) backends such as Ollama.

## Core Types

| Type | Description |
|------|-------------|
| `ResponseRequest` | Input payload containing the raw text, NLP result, and dialogue context. |
| `ResponsePlan` | Wraps the request plus tone/persona guidance derived from sentiment analysis. |
| `ResponseCandidate` | A generated reply annotated with strategy metadata. |
| `StyleGuide` | Tone, voice, and escalation hints for the responder. |

## ResponseGenerator

```python
from lyra import ResponseGenerator

generator = ResponseGenerator()
reply = generator.generate("Hey Lyra, how are we doing on the deployment?", session_id="demo")
print(reply.text)
```

### Responders

- `TemplateResponder` (default): deterministic responses for greetings, status
  checks, shutdown requests, etc. It leverages sentiment tone hints to adjust
  empathy or escalate when needed.
- `OllamaResponder`: optional LLM-backed responder that calls a local Ollama
  server (`http://localhost:11434/api/generate`) using the configured model
  name (default `llama3`). Enable it by constructing:

```python
from lyra.response import OllamaResponder
generator = ResponseGenerator(responders=[OllamaResponder(enabled=True), TemplateResponder()])
```

If the Ollama server is unreachable, the generator falls back to the next
strategy (typically the template responder).

### Dynamic Tone

The tone metadata in `ResponsePlan.guidance` comes directly from the sentiment
module. This ensures responses remain empathetic during negative interactions
and upbeat when the conversation is positive. The dialogue manager stores the
latest tone hints inside the short-term context so future turns can adjust
even if a different responder is used. The `StyleGuide.apply()` helper adds
emotion-aware prefixes/suffixes (e.g., empathy statements for negative tone or
celebratory language for positive tone) before the candidate is returned.

### Extending

Implement custom responders by subclassing `BaseResponder` and registering them
via `ResponseGenerator.register_responder(responder, priority=0)`. Responders
receive the full `ResponsePlan`, enabling advanced use cases such as:

- Querying external knowledge bases before replying.
- Streaming responses from transformer pipelines (e.g., Hugging Face models).
- Applying domain-specific guardrails prior to returning text.


