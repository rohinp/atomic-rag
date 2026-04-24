# Multi-Query Expansion

## Problem Statement

A single query phrasing can miss relevant documents that express the same concept differently:

- User asks: *"authentication flow"*
- Relevant chunk 1 uses: *"token validation pipeline"*
- Relevant chunk 2 uses: *"login sequence"*
- Relevant chunk 3 uses: *"how the API verifies credentials"*

A vector search on the original query may rank all three poorly, not because the embedder is bad, but because the query occupies one point in the embedding space while the documents spread across nearby but distinct regions. Keyword search (BM25) has the same problem — none of the chunks contain the exact words "authentication flow".

## How It Works

1. **Generate**: send the query to an LLM, asking it to produce N alternative phrasings.
2. **Retrieve**: run vector search independently for each phrasing (including the original).
3. **Fuse**: merge all N result lists with Reciprocal Rank Fusion.

```
query: "authentication flow"
  │
  │ LLM.complete(prompt)
  ▼
alternatives:
  - "token validation pipeline"
  - "how does user login work"
  - "credential verification sequence"
  │
  ├─ embed + vector search → ranked list 1
  ├─ embed + vector search → ranked list 2
  ├─ embed + vector search → ranked list 3
  └─ BM25 on original     → ranked list 4
                                │
                          RRF fusion
                                │
                         final ranked list
```

Documents that appear highly ranked in *any* list get boosted by RRF. A document about "token validation" that ranks #1 for query 2 will appear near the top of the fused list even if it was rank #25 for the original query.

## Developer Benefits

- **Coverage**: different phrasings probe different regions of the embedding space, catching documents that any single query would miss.
- **No training required**: the LLM generates alternatives at inference time — no labelled data, no fine-tuning.
- **Composable with HyDE**: can be used together (HyDE generates a hypothetical doc, MultiQuery generates alternatives) or independently.
- **Transparent to downstream**: just populates `expanded_queries`; HybridRetriever consumes it without any API change.
- **Observable**: TraceEntry records `requested`, `generated`, and `expanded_count` per call.
- **Graceful fallback**: on LLM failure, returns the original query so retrieval still works.

## Alternatives

| Alternative | Trade-offs |
|---|---|
| **Single query** (baseline) | Zero overhead. Fine for well-specified queries against tight corpora. Misses vocabulary variance. |
| **HyDE** | One hypothetical document rather than alternative questions. Better for short, underspecified queries. Complementary, not competing. |
| **Synonym expansion** (WordNet/NLTK) | Rule-based synonym injection, no LLM needed. Fast but brittle — synonyms aren't context-aware. "Bark" the tree vs. "bark" the dog. |
| **Query reformulation via fine-tuned model** | Small encoder-decoder trained specifically for query rewriting. Higher quality but requires training data. |
| **Dense passage retrieval (DPR) fine-tuning** | Train the embedder to handle query variance. Requires labelled data and GPU. |

## Effectiveness

Multi-query retrieval is a well-established practitioner technique. Quantitative gains depend on corpus vocabulary diversity:

- On corpora with high vocabulary variance (e.g. open-domain QA, conversational queries), multi-query retrieval consistently outperforms single-query by **5–15% Recall@K**.
- On tightly scoped technical corpora (e.g. API docs where queries and docs share terminology), gains are smaller — 1–5%.
- Combining with RRF (vs. score averaging) ensures that documents retrieved by any single expansion candidate are surfaced even if other candidates rank them poorly.

**Caveats**:

- Adds N LLM inference calls per query (or 1 call that generates N alternatives). With a local 7B model this can add 2–5 seconds.
- If the LLM generates low-quality alternatives (e.g. very similar to the original), the benefit diminishes. Inspect trace `generated` counts and review expansion quality in development.
- N=3 is a reasonable default. Beyond N=5, marginal gain drops sharply while latency grows linearly.

## Usage in atomic-rag

```python
from atomic_rag.query import MultiQueryExpander
from atomic_rag.models.ollama import OllamaChat
from atomic_rag.schema import DataPacket

expander = MultiQueryExpander(
    chat_model=OllamaChat(model="llama3.2:3b"),
    n_queries=3,                 # 3 alternatives + original = 4 total
)

packet = DataPacket(query="authentication flow")
packet = expander.expand(packet)
# packet.expanded_queries = ["authentication flow", "token validation pipeline", ...]

# HybridRetriever uses expanded_queries automatically
packet = retriever.retrieve(packet, top_k=5)
```

Exclude the original query:

```python
expander = MultiQueryExpander(
    chat_model=OllamaChat(),
    n_queries=4,
    include_original=False,  # alternatives only
)
```

Combine with HyDE:

```python
# HyDE first, then multi-query on top — maximum coverage
hyde = HyDEExpander(chat_model=OllamaChat())
packet = hyde.expand(packet)
# packet.expanded_queries = [hypothetical_doc]

# Now also add original + alternatives
multi = MultiQueryExpander(chat_model=OllamaChat(), n_queries=2, include_original=True)
packet = multi.expand(packet)
# packet.expanded_queries = [original, alt1, alt2]
# Note: running both is redundant unless you fuse their expanded_queries manually
```

> Tip: for most use cases, pick one strategy. HyDE for short/vague queries; MultiQuery for queries where vocabulary variance is the main problem.

---

## Research

**Jagerman et al. (2023). "Query Expansion by Prompting Large Language Models."**
[arXiv:2305.03653](https://arxiv.org/abs/2305.03653)

Shows that LLM-generated query expansions outperform classical pseudo-relevance feedback on several BEIR benchmarks, with the largest gains on queries where vocabulary mismatch between the question and the relevant documents is highest.

| Claim verified by test | Test |
|---|---|
| N alternative phrasings generated | [`test_query.py → TestMultiQueryExpander::test_n_queries_limits_alternatives`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Original query included by default | [`test_query.py → TestMultiQueryExpander::test_original_included_by_default`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Multiple embed calls issued — one per query | [`test_query.py → TestHybridRetrieverQueryIntegration::test_multiple_embed_calls_with_expanded_queries`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Trace records requested vs generated count | [`test_query.py → TestMultiQueryExpander::test_trace_records_requested_and_generated`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |

→ [Full reference list](../references.md)
