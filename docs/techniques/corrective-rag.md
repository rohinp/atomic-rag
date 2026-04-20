# Corrective RAG (C-RAG)

## Problem Statement

Standard RAG pipelines always generate an answer — even when the retrieved context is irrelevant, empty, or about a completely different topic. The result is a confident-sounding hallucination:

- User asks: *"What was Q4 cloud revenue?"*
- Retrieved chunk: *"The company was founded in 2001 and has operations in 40 countries."*
- Standard RAG response: *"Q4 cloud revenue was approximately $2.4B, reflecting strong growth..."* (fabricated)

The LLM fills the gap between what the context says and what the question asks. Without a quality gate, it will always produce *something* — and wrong with confidence is worse than an honest "I don't know."

**Concrete example of the failure**: a technical Q&A system where the retriever returns a doc about the wrong function, but the generator produces a plausible-looking (wrong) API description anyway.

## How It Works

C-RAG inserts an **evaluator** between retrieval and generation:

```
context (Phase 4 output)
         │
         ▼
    LLMEvaluator ──── score in [0.0, 1.0]
         │
     score >= threshold?
        /           \
      YES             NO
       │               │
  LLMGenerator    fallback message
       │               │
    answer         "I don't have enough
                    information..."
```

1. **Evaluate**: The evaluator asks a grading LLM: *"Does this context answer the question? Score 0.0–1.0."* The score reflects how completely the retrieved context addresses the query.

2. **Route**:
   - Score ≥ threshold → generate a grounded answer from the context
   - Score < threshold → return a configurable fallback message (no generation)

3. **Generate**: The generator prompts the LLM with context + query and instructs it to stay within the context, citing source labels where available.

**Why a separate evaluator call instead of instructing the generator to refuse?** Instruction-following for "refuse if context is insufficient" is unreliable, especially in smaller models. An explicit evaluator with a numeric score gives a deterministic, observable gate. You can log and tune the threshold rather than hoping the model behaves.

## Developer Benefits

- **Eliminates confident hallucinations**: the system says "I don't know" honestly rather than fabricating an answer.
- **Threshold is observable**: `TraceEntry.details` records `eval_score`, `threshold`, and `fallback` per call. Tune the threshold by inspecting traces in development.
- **Generator not called on fallback**: avoids LLM inference cost when context is irrelevant.
- **Composable**: evaluator and generator are injected dependencies — swap either without touching `AgentRunner`.
- **Source-cited answers**: the generator prompt instructs the LLM to cite `[Source: ...]` labels from Phase 4 output, enabling downstream citation extraction.

## Alternatives

| Alternative | Trade-offs |
|---|---|
| **No quality gate** | Simplest. Acceptable when the corpus is curated and retrieval is reliable. Fails when queries go out of distribution. |
| **Instruction-based refusal** ("If context is insufficient, say 'I don't know'") | No extra LLM call. Unreliable with smaller models — they often comply only partially. |
| **Perplexity-based filtering** | Use the generator's own perplexity on the context as a proxy for relevance. No extra call, but hard to threshold reliably across different queries. |
| **SELF-RAG** (Asai et al., 2023) | Uses special reflection tokens to decide mid-generation whether to retrieve more. More powerful but requires fine-tuning the generator. |
| **Fallback to web search** | Trigger a live web search when context is insufficient. Higher recall but adds latency, external dependency, and new hallucination risk from web content. |

## Effectiveness

Yan et al., "Corrective Retrieval Augmented Generation" (2024):

- C-RAG outperforms standard RAG on 4 of 4 knowledge-intensive QA benchmarks (PopQA, Arc-Challenge, PubHealth, Biography).
- On PopQA: standard RAG EM = 38.2, C-RAG = 43.9 (+5.7 points).
- Biggest gains occur on out-of-distribution queries where the retriever frequently returns low-relevance documents.

**Caveats**:

- Adds one evaluator LLM call per query (typically 50–200ms with a local 3B model). Total pipeline latency is roughly doubled vs. retrieval-only.
- Threshold tuning is corpus-dependent. Start at 0.5. If `fallback=True` fires too often on valid queries, lower it. If hallucinations still occur, raise it.
- The evaluator itself can make mistakes — if the grading LLM is smaller than the generator, it may fail to recognise relevant context. Monitor `eval_score` distributions in traces.

## Usage in atomic-rag

### DataPacket contract

**Input**: `DataPacket` with `query` and `context` set (Phase 4 output).

**Output**: New `DataPacket` with:
- `answer` — generated answer or fallback message
- `trace` — one `TraceEntry` appended, `phase="agent"`

### Minimal example

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

# packet has query + context from Phases 3–4
result = runner.run(packet)

print(result.answer)
print(result.trace[-1].details)
# {'eval_score': 0.82, 'threshold': 0.5, 'fallback': False, 'answer_length': 143}
```

### Tuning threshold

```python
# More permissive — generate even with weak context (lower hallucination protection)
runner = AgentRunner(evaluator=evaluator, generator=generator, threshold=0.3)

# Stricter — only generate when context is clearly sufficient
runner = AgentRunner(evaluator=evaluator, generator=generator, threshold=0.7)

# Custom fallback message
runner = AgentRunner(
    evaluator=evaluator,
    generator=generator,
    fallback_message="The knowledge base doesn't contain information on this topic.",
)
```

### Using different models for evaluator and generator

```python
from atomic_rag.models.openai_provider import OpenAIChat

# Small/fast model for evaluation, stronger model for generation
runner = AgentRunner(
    evaluator=LLMEvaluator(chat_model=OllamaChat(model="llama3.2:3b")),
    generator=LLMGenerator(chat_model=OpenAIChat(model="gpt-4o-mini")),
)
```
