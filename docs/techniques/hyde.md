# HyDE — Hypothetical Document Embeddings

## Problem Statement

Embedding-based retrieval compares a query vector against document chunk vectors using cosine similarity. This works well when the query and the document use the same vocabulary and writing style. It breaks down when they diverge:

- User asks: *"auth flow?"* (4 words, casual)
- Relevant document says: *"The authentication pipeline validates the JWT token against the public key stored in the secrets manager, then issues a short-lived session credential..."* (technical, verbose)

Even if the embedding model understands both, the **geometric distance** between a 4-word query vector and a 200-word paragraph vector is large. The query vector is underspecified — it occupies a vague region of the embedding space that doesn't cleanly overlap with any specific document cluster.

## How It Works

1. **Generate**: Send the query to an LLM with a prompt like: *"Write a short passage that directly answers this question."*
2. **Embed**: Embed the *hypothetical document* (not the original query).
3. **Retrieve**: Use that embedding for vector search.

The hypothetical document is long, domain-specific, and written in the same register as real documents. It occupies a tight, discriminating region of the embedding space — much closer to the actual relevant passages than a 4-word query would be.

**Why this works**: LLMs trained on domain corpora write hypothetical answers using the same vocabulary and style as real documents. The embedding of "Cloud revenue grew 34% to $1.6B in Q4, driven by infrastructure services" is geometrically close to the real chunk that says almost the same thing — even if the user only asked *"Q4 cloud results?"*

**Diagram**:

```
query: "Q4 cloud results?"
  │
  │ LLM.complete(prompt)
  ▼
hypothetical: "Cloud revenue grew significantly in Q4 2024,
               reaching $1.6B..."
  │
  │ embedder.embed(hypothetical)
  ▼
[vector search] → relevant chunk found at rank 1
```

## Developer Benefits

- **No vocabulary bridging required**: the LLM handles the translation from casual user language to domain terminology automatically.
- **Single inference call**: one LLM call per query, parallelisable with other retrieval stages.
- **Composable**: just populates `expanded_queries` in the DataPacket; HybridRetriever uses it transparently. No changes to the retriever.
- **Measurable**: `TraceEntry.details` records `hypothetical_length` for monitoring.
- **Graceful fallback**: on LLM failure, falls back to the original query — retrieval degrades gracefully rather than crashing.

## Alternatives

| Alternative | Trade-offs |
|---|---|
| **Raw query embedding** (baseline) | Zero latency overhead, but poor on short queries or vocabulary mismatch. Good enough for well-structured, terminology-matched corpora. |
| **Query rewriting** (without HyDE) | Rephrase the query for clarity without generating a full document. Cheaper, but the resulting query is still short and may underspecify the embedding. |
| **Multi-Query Expansion** | Generate alternative question phrasings instead of a hypothetical document. Better coverage; complementary to HyDE (they can be combined). |
| **Dense retrieval fine-tuning** (e.g. ANCE) | Train the embedder to handle short queries better. High quality but requires labelled data and a GPU fine-tuning run. |

## Effectiveness

Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022):

- HyDE outperforms standard dense retrieval on 11 of 11 BEIR benchmarks in zero-shot settings.
- Particularly large gains on low-resource domains where queries are short and vocabulary gap is wide.
- On MS-MARCO (passage retrieval), HyDE achieves nDCG@10 of **56.6** vs **43.3** for DPR without fine-tuning.

**Caveats**:

- Adds one LLM inference call per query. If the LLM is slow (e.g. local 7B model), this adds latency. Use a fast/small model for hypothesis generation.
- If the LLM generates a confidently wrong hypothesis, the embedding will search in the wrong region. For high-stakes retrieval, combine with Multi-Query Expansion so the original query also gets searched.
- Less benefit when chunks are short (≤ 3 sentences) and vocabularies already overlap — baseline embedding is often sufficient in those cases.

## Usage in atomic-rag

```python
from atomic_rag.query import HyDEExpander
from atomic_rag.models.ollama import OllamaChat
from atomic_rag.schema import DataPacket

expander = HyDEExpander(chat_model=OllamaChat(model="llama3.2:3b"))

packet = DataPacket(query="how does authentication work?")
packet = expander.expand(packet)  # expanded_queries = [hypothetical_doc]

# Pass to HybridRetriever — it uses expanded_queries automatically
packet = retriever.retrieve(packet, top_k=5)
```

Fallback on LLM error:

```python
# Default: fall back to original query on LLM failure
expander = HyDEExpander(chat_model=OllamaChat(), fallback_to_original=True)

# Strict: raise on LLM failure
expander = HyDEExpander(chat_model=OllamaChat(), fallback_to_original=False)
```

Custom prompt template:

```python
template = (
    "You are a technical writer. Write a 3-sentence explanation that answers: {query}\n"
    "Use precise technical language."
)
expander = HyDEExpander(chat_model=OllamaChat(), prompt_template=template)
```


---

## Research

**Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels."**
[arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

The paper that introduced HyDE. Key finding: embedding a *hypothetical answer document* generated by an LLM outperforms embedding the raw query on BEIR zero-shot retrieval benchmarks, because the hypothesis lives in the same semantic space as real answer documents.

| Claim verified by test | Test |
|---|---|
| Hypothetical doc populates `expanded_queries` | [`test_query.py → TestHyDEExpander::test_populates_expanded_queries`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Exactly one expanded query produced | [`test_query.py → TestHyDEExpander::test_expanded_queries_has_exactly_one_entry`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Graceful fallback to original on LLM error | [`test_query.py → TestHyDEExpander::test_fallback_to_original_on_error`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Trace records hypothetical document length | [`test_query.py → TestHyDEExpander::test_trace_records_hypothetical_length`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |

→ [Full reference list](../references.md)
