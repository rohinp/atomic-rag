# Cross-Encoder Reranking

## Problem Statement

First-stage retrieval (vector search, BM25) encodes the query and documents *independently* — the query vector is computed once, then compared against pre-stored document vectors. This is fast enough to search millions of documents in milliseconds, but it is inherently limited: the model never sees the query and document *together*, so it cannot model fine-grained relevance interactions.

The result is that the top-5 from first-stage retrieval often contains irrelevant documents that happened to be close in the embedding space, while the truly relevant document sits at rank 8 or 15 — just outside what the LLM will see.

**Concrete example**: Query `"how do I handle a timeout in the connection pool?"` may retrieve documents about connection pools generally, but the specific document describing timeout configuration might rank lower because "timeout" is a common word that the embedder doesn't emphasize.

## How It Works

A cross-encoder is a BERT-style model that takes the **concatenated** query and document as input and outputs a single relevance score. Because it reads both at the same time, it can model interactions like:

- "timeout" appearing in the context of `ConnectionPool` is more relevant than "timeout" in a logging module
- A document that never uses the word "pool" but describes the exact timeout configuration is still highly relevant

The cross-encoder is applied only to the small set of candidates (typically 50) that survived first-stage retrieval. This two-stage design keeps latency manageable:

```
first stage:  ~1,000,000 docs → 50 candidates  (milliseconds, ANN)
second stage: 50 candidates  →  5 results       (100–500ms, cross-encoder)
```

Internally:
1. Each (query, document) pair is tokenised and passed through the model
2. The model outputs a logit (real number, no fixed range)
3. Documents are sorted by logit descending — higher = more relevant
4. Top-k are returned with `Document.score` set to the logit

The score is **not a probability** — it is only meaningful for ordering, not as an absolute threshold.

## Developer Benefits

- **Precision at the top**: Moves the most relevant document from rank 8 to rank 1–2. For a 5-document context window, this is the difference between a correct answer and a hallucinated one.
- **Pluggable**: Any HuggingFace cross-encoder works as a drop-in replacement. The interface is model-agnostic.
- **Cacheable at index time**: The document embeddings from first-stage retrieval are computed once. Only the cross-encoder inference (50 pairs) happens per query.
- **Observable**: Before/after scores are visible in `Document.score` and `TraceEntry.details`. You can log both first-stage and reranked rankings to debug retrieval quality.

## Alternatives

| Alternative | Trade-offs |
|---|---|
| **No reranking** (RRF only) | Faster, simpler. Adequate if first-stage recall is already high (e.g. small, well-structured corpus). Missing the ~10–20% precision gain of reranking. |
| **MonoT5 reranker** | T5-based reranker, stronger than MiniLM cross-encoders. Larger, slower. Use if quality is more important than latency. |
| **BAAI/bge-reranker-v2-m3** | Multilingual, strong on technical content and code. Good alternative to ms-marco models for non-English or code-heavy corpora. |
| **Cohere Rerank API** | State-of-the-art quality, API-based. No local inference. Adds network latency and per-call cost. Best for production where quality matters most. |
| **Voyage AI rerank-2** | High quality, especially strong on code. API-based. |
| **ColBERT** | Avoids the two-stage problem entirely via late-interaction: stores token-level embeddings and scores at query time. Very high recall but requires specialised indexing infrastructure. |

**When to skip reranking**: If your corpus is small (< 1,000 documents), first-stage retrieval quality is usually sufficient. Add a reranker when you observe that the correct document is in the retrieved set but not at the top — visible by logging `packet.documents` before and after.

## Effectiveness

From MS MARCO Passage Ranking (the standard benchmark):

| Model | MRR@10 |
|---|---|
| BM25 (baseline) | 0.187 |
| Dense retrieval (bi-encoder) | ~0.33 |
| Dense + cross-encoder reranker | ~0.39 |
| Dense + MonoT5 reranker | ~0.42 |

The cross-encoder consistently adds **+5–10% MRR** over first-stage dense retrieval alone on general benchmarks. On technical and code-specific retrieval tasks, gains are often larger because the cross-encoder can model identifier co-occurrence that bi-encoders miss.

**Latency**: `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU: ~50–150ms for 50 pairs. On GPU: ~10–30ms.

**Caveats**:
- Cross-encoders are sensitive to document length. Very long chunks may be truncated (512 token limit for most BERT models). Keep `max_chunk_chars` ≤ 1,500 characters to stay within limits.
- The score is a logit, not a probability. Do not use it as a relevance threshold (e.g. `if score > 0.5`). Only use it for ranking.

## Usage in atomic-rag

### DataPacket contract

`CrossEncoderReranker` is not a `RetrieverBase` — it is called internally by `HybridRetriever` and does not interact with `DataPacket` directly. It takes `(query, documents, top_k)` and returns a reordered `list[Document]` with updated scores.

### Minimal example

```python
from atomic_rag.retrieval import CrossEncoderReranker
from atomic_rag.schema import Document

# Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
reranker = CrossEncoderReranker()

# After first-stage retrieval...
candidates: list[Document] = [...]  # 50 candidates from HybridRetriever

top5 = reranker.rerank(query="timeout in connection pool", documents=candidates, top_k=5)
```

### Via HybridRetriever

```python
from atomic_rag.retrieval import CrossEncoderReranker, HybridRetriever
from atomic_rag.models.ollama import OllamaEmbedder

retriever = HybridRetriever(
    embedder=OllamaEmbedder(),
    reranker=CrossEncoderReranker(model="BAAI/bge-reranker-base"),
)
```

### Swapping the model

```python
# Stronger, slower
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-electra-base")

# Multilingual + code
reranker = CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")
```
