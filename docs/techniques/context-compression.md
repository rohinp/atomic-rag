# Context Compression

## Problem Statement

After retrieval, the top-k documents are concatenated and fed to the LLM. Each document chunk may be 200–500 words, but only 2–3 sentences in it are actually relevant to the query. The rest is padding — background context, related-but-not-answering sentences, boilerplate — and it has a measurable effect:

**The "Lost in the Middle" problem** (Liu et al., 2023): when relevant information is surrounded by irrelevant text in a long context window, LLMs consistently perform worse than when the relevant information is presented cleanly. The model attends strongly to the beginning and end of the context but misses content buried in the middle.

**Concrete example**: Query: *"What was cloud revenue in Q4?"* Retrieved chunk:

> The company was founded in 2001 and has operations across 40 countries. In Q4 2024, cloud revenue grew 34% year-over-year to $1.6 billion. The CEO commented on the strong performance during the earnings call. The company plans to expand its data centre footprint in 2025. Employee headcount reached 52,000.

Only one sentence is relevant. Sending all five wastes tokens and dilutes the signal. If this pattern repeats across 5 retrieved chunks, the LLM receives ~25 sentences but only ~5 are about the query.

## How It Works

For each retrieved `Document`, the compressor:

1. **Splits** content into segments. Prose → sentence boundaries (regex). Code → the whole chunk (splitting a function body would make it syntactically broken and semantically useless).
2. **Embeds** all segments in a single batch call against the same embedder used for retrieval — keeps the vector space consistent.
3. **Scores** each segment via cosine similarity to the query embedding.
4. **Keeps** segments at or above `threshold`. Always keeps at least `min_sentences` per document (the highest-scoring ones) so no document disappears entirely.
5. **Assembles** the filtered segments back into a context string with source labels so the LLM can cite accurately.

The output format:

```
[Source: src/auth/service.py  L42–61]
def authenticate(token: str) -> User: ...

---

[Source: reports/q4-2024.pdf  chunk 3]
In Q4 2024, cloud revenue grew 34% year-over-year to $1.6 billion.
```

The source labels are critical — without them, the LLM cannot cite where a claim came from and is more likely to hallucinate a file path.

**Why the same embedder?** Using a different model for sentence scoring than for document retrieval creates an inconsistency — a sentence might score high by one model's geometry but low by another's. Reusing the retrieval embedder guarantees that "high similarity to the query" means the same thing at both stages.

## Developer Benefits

- **Fewer tokens**: Compression typically removes 30–60% of the retrieved text, directly reducing LLM API cost and latency.
- **Better answer quality**: Removing low-signal sentences reduces "Lost in the Middle" failures, particularly for multi-document contexts.
- **Observable**: `TraceEntry.details` records `sentences_before`, `sentences_after`, and `reduction_pct` per call. Tune `threshold` by inspecting these in Langfuse traces.
- **Code-safe**: Code chunks are never split. A partial function in the context would confuse the LLM; the compressor detects `metadata["language"]` and keeps code whole.
- **No extra dependency**: Uses the same embedder already in the pipeline. No new model or API key required.

## Alternatives

| Alternative | Trade-offs |
|---|---|
| **No compression** (pass full chunks) | Simplest. Acceptable for small corpora where chunks are already tight. Fails at scale when chunks are verbose. |
| **Token truncation** (hard cut at N tokens) | Fast but blunt — cuts the most relevant sentence if it's at the end of the chunk. |
| **Extractive summarisation** (e.g. LexRank, TextRank) | Graph-based sentence scoring, no LLM needed. Slower, more complex, similar quality to cosine filtering for RAG contexts. |
| **LLM-based compression** (e.g. LLMLingua, Selective Context) | Uses a small LM to compress token-by-token. Higher quality but adds latency (second model inference per document). Use when sentence-level filtering is too coarse. |
| **LLMLingua-2** | Distilled compressor, faster than original LLMLingua, strong benchmark results. API-based option available. Worth evaluating when sentence compression is insufficient. |
| **Reranker-only** (skip Phase 4, rely on Phase 3b) | If cross-encoder reranking already surfaces only highly-relevant chunks, compression adds less value. Evaluate by comparing answer quality with and without it. |

**When to skip compression**: If your chunks are already small (≤ 3 sentences each) and tightly scoped (e.g. one function = one chunk), compression adds latency without meaningful benefit. Enable it for prose-heavy document corpora.

## Effectiveness

Liu et al., "Lost in the Middle" (2023) — the foundational study:
- Performance on multi-document QA drops significantly as context length increases and relevant documents are placed in the middle of the window.
- Models answered correctly 75% of the time when the relevant document was at position 1, but only ~45% when it was at position 10 in a 20-document context.

Context compression directly addresses this by ensuring the context window contains primarily relevant text, reducing the chance of the key fact ending up "in the middle".

On practical benchmarks:
- Sentence-level cosine filtering typically achieves **30–60% token reduction** with negligible answer quality loss (sometimes improvement) on question-answering tasks.
- Threshold tuning matters: too high (> 0.7) can drop relevant sentences that use different phrasing; too low (< 0.3) provides minimal filtering.

**Caveats**:
- The threshold is corpus-dependent. Start at 0.5, inspect `reduction_pct` in traces, and adjust.
- Does not work well for queries that are very short (1–2 words) because the query embedding is too underspecified to score sentences accurately.

## Usage in atomic-rag

### DataPacket contract

**Input**: `DataPacket` with `query` and `documents` set (Phase 3 output).

**Output**: New `DataPacket` with:
- `context`: compressed string with source labels, ready for the LLM
- `trace`: one `TraceEntry` appended, `phase="context"`

### Minimal example

```python
from atomic_rag.context import SentenceCompressor
from atomic_rag.models.ollama import OllamaEmbedder
from atomic_rag.schema import DataPacket

# After Phase 3 retrieval...
compressor = SentenceCompressor(embedder=OllamaEmbedder(), threshold=0.5)
result = compressor.compress(packet)  # packet has documents from retrieval

print(result.context)        # compressed, labelled context for the LLM
print(result.trace[-1].details)  # {"reduction_pct": 45.2, "sentences_before": 22, ...}
```

### Tuning threshold

```python
# Stricter — more aggressive filtering, smaller context, higher risk of dropping relevant sentences
compressor = SentenceCompressor(embedder=OllamaEmbedder(), threshold=0.65)

# More permissive — keeps more context, safer for short or ambiguous queries
compressor = SentenceCompressor(embedder=OllamaEmbedder(), threshold=0.35)

# Always keep at least 2 sentences per document
compressor = SentenceCompressor(embedder=OllamaEmbedder(), threshold=0.6, min_sentences=2)
```
