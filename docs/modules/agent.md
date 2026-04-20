# Agent Module

**Location**: `atomic_rag/agent/`

Implements Phase 5: Corrective RAG (C-RAG) — evaluates context quality before generation to prevent hallucinations, then produces a grounded answer from the retrieved context.

## Components

| Class | Role |
|---|---|
| `EvaluatorBase` | Abstract interface — subclass to add a custom context quality scorer |
| `GeneratorBase` | Abstract interface — subclass to add a custom answer generator |
| `LLMEvaluator` | Uses an LLM to score how well the context answers the query (0.0–1.0) |
| `LLMGenerator` | Uses an LLM to produce a grounded answer from query + context |
| `AgentRunner` | Orchestrates the C-RAG loop: evaluate → generate or fallback |

See [Corrective RAG](../techniques/corrective-rag.md) for the research rationale, alternatives, and effectiveness data.

## DataPacket Contract

**Input**: `DataPacket` with `query` and `context` set (Phase 4 output).

**Output**: New `DataPacket` with:
- `answer` — generated answer string, or fallback message if context quality is insufficient
- `trace` — one `TraceEntry` appended (`phase="agent"`)

Input packet is never mutated.

## Minimal Working Example

```python
from atomic_rag.agent import AgentRunner, LLMEvaluator, LLMGenerator
from atomic_rag.models.ollama import OllamaChat
from atomic_rag.schema import DataPacket

chat = OllamaChat(model="llama3.2:3b")

runner = AgentRunner(
    evaluator=LLMEvaluator(chat_model=chat),
    generator=LLMGenerator(chat_model=chat),
    threshold=0.5,
)

# packet.context comes from Phase 4 (SentenceCompressor)
result = runner.run(packet)

print(result.answer)
print(result.trace[-1].details)
# {'eval_score': 0.82, 'threshold': 0.5, 'fallback': False, 'answer_length': 143}
```

## Configuration

### AgentRunner

| Parameter | Default | Effect |
|---|---|---|
| `evaluator` | required | Any `EvaluatorBase` implementation |
| `generator` | required | Any `GeneratorBase` implementation |
| `threshold` | `0.5` | Minimum eval score to proceed with generation; below this, fallback fires |
| `fallback_message` | built-in | Returned as `answer` when context is insufficient |

### LLMEvaluator

| Parameter | Default | Effect |
|---|---|---|
| `chat_model` | required | Any `ChatModelBase` |
| `prompt_template` | built-in | Must contain `{query}` and `{context}` placeholders |
| `default_score` | `0.5` | Score returned when LLM output cannot be parsed as a float |

Score parsing handles: plain floats (`0.8`), embedded floats (`Score: 0.7`), YES → 1.0, NO → 0.0.
Empty or whitespace-only context always returns `0.0` without calling the LLM.

### LLMGenerator

| Parameter | Default | Effect |
|---|---|---|
| `chat_model` | required | Any `ChatModelBase` |
| `prompt_template` | built-in | Must contain `{query}` and `{context}` placeholders |

## Swapping the Backend

Any `EvaluatorBase` or `GeneratorBase` can be replaced without touching `AgentRunner`:

```python
class ThresholdEvaluator(EvaluatorBase):
    """Approve if context is non-empty and long enough."""
    def evaluate(self, query: str, context: str) -> float:
        return 1.0 if len(context) > 100 else 0.0

runner = AgentRunner(
    evaluator=ThresholdEvaluator(),
    generator=LLMGenerator(chat_model=chat),
)
```

Use a stronger model for generation and a smaller model for evaluation:

```python
runner = AgentRunner(
    evaluator=LLMEvaluator(chat_model=OllamaChat(model="llama3.2:3b")),
    generator=LLMGenerator(chat_model=OpenAIChat(model="gpt-4o-mini")),
)
```

## Running Tests

```bash
pytest tests/test_agent.py -v
```
