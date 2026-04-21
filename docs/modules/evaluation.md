# Evaluation Module

**Location**: `atomic_rag/evaluation/`

Measures answer quality and retrieval quality after the full pipeline has run. Evaluators are composable — run one or several; each populates only the `eval_scores` field it owns without disturbing others.

## Components

| Class | Metric computed | Requires |
|---|---|---|
| `PipelineEvalBase` | Abstract interface | — |
| `LLMFaithfulnessScorer` | `eval_scores.faithfulness` | ChatModelBase |
| `EmbeddingAnswerRelevance` | `eval_scores.answer_relevance` | EmbedderBase |
| `RagasEvaluator` | All Ragas metrics → EvalScores | `ragas>=0.2`, `datasets` |

## DataPacket Contract

**Input**: `DataPacket` with `query`, `context`, and `answer` set (Phase 5 output).

**Output**: New `DataPacket` with:
- `eval_scores` — one or more fields populated (others left unchanged)
- `trace` — one `TraceEntry` appended per evaluator (`phase="evaluation"`)

Evaluators can be chained: run `LLMFaithfulnessScorer` then `EmbeddingAnswerRelevance` on the same packet and both fields will be set.

## Metrics Reference

| Metric | Field | Range | Meaning |
|---|---|---|---|
| Faithfulness | `eval_scores.faithfulness` | 0–1 | Fraction of answer claims supported by the context. 0 = hallucinated, 1 = fully grounded. |
| Answer relevance | `eval_scores.answer_relevance` | 0–1 | Cosine similarity between query and answer embeddings. 1 = answer directly addresses the question. |
| Context precision | `eval_scores.context_precision` | 0–1 | Ragas metric: is the relevant document ranked highly? Requires `reference`. |

## Minimal Working Example

```python
from atomic_rag.evaluation import LLMFaithfulnessScorer, EmbeddingAnswerRelevance
from atomic_rag.models.ollama import OllamaChat, OllamaEmbedder

# packet.query, packet.context, packet.answer must be set (after Phase 5)

faith_scorer = LLMFaithfulnessScorer(chat_model=OllamaChat(model="llama3.2:3b"))
rel_scorer = EmbeddingAnswerRelevance(embedder=OllamaEmbedder())

packet = faith_scorer.score(packet)
packet = rel_scorer.score(packet)

print(packet.eval_scores.faithfulness)    # e.g. 0.83
print(packet.eval_scores.answer_relevance)  # e.g. 0.91
print(packet.trace[-2].details)  # faithfulness trace
print(packet.trace[-1].details)  # answer_relevance trace
```

## Using Ragas

Install extras first:

```bash
pip install 'ragas>=0.2' datasets langchain-ollama
# or for OpenAI:
pip install 'ragas>=0.2' datasets langchain-openai
```

Configure Ragas with an LLM and run:

```python
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = LangchainLLMWrapper(ChatOllama(model="llama3.2:3b"))
emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

metrics = [
    Faithfulness(llm=llm),
    AnswerRelevancy(llm=llm, embeddings=emb),
]

from atomic_rag.evaluation import RagasEvaluator

evaluator = RagasEvaluator(metrics=metrics)
result = evaluator.score(packet)

print(result.eval_scores.faithfulness)
print(result.eval_scores.answer_relevance)
print(result.trace[-1].details)
# {'evaluator': 'ragas', 'metrics': ['Faithfulness', 'AnswerRelevancy'], 'scores': {...}}
```

For `ContextPrecision` (requires a reference answer):

```python
from ragas.metrics import ContextPrecision

evaluator = RagasEvaluator(
    metrics=[ContextPrecision(llm=llm)],
    reference="The ground-truth answer goes here.",
)
result = evaluator.score(packet)
print(result.eval_scores.context_precision)
```

## Choosing an Evaluator

| Situation | Recommendation |
|---|---|
| Quick offline scoring, no extra deps | `LLMFaithfulnessScorer` + `EmbeddingAnswerRelevance` |
| Benchmark against labelled dataset | `RagasEvaluator` with `ContextPrecision` + `reference` |
| Minimal latency (embedding only) | `EmbeddingAnswerRelevance` only |
| Maximum metric coverage | `RagasEvaluator` with all relevant metrics |

## Running Tests

```bash
pytest tests/test_evaluation.py -v
```
