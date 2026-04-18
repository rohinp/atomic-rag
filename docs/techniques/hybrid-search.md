# Hybrid Search (Vector + BM25)

## Problem Statement

Vector search embeds both the query and each document into a dense vector space and retrieves the nearest neighbours. This works well for semantic similarity — "how do I authenticate a user?" correctly matches documents about auth even if they use different terminology.

But it fails on exact terms. A query for `ConnectionPool` may not retrieve the document defining that class if the embedding model smooths it into the same neighbourhood as `connection`, `pool`, and `network socket`. Acronyms, version numbers, function names, and error codes all suffer the same fate.

BM25 has the opposite profile: it matches exact tokens precisely, but cannot bridge semantic gaps. A query for "authentication" would miss a document that only uses the word "login".

Neither alone is sufficient. They fail on complementary cases.

## How It Works

**Stage 1: Parallel retrieval**

Both retrievers run independently against the same document corpus, each returning a ranked list of `candidate_k` results (typically 50):

- **Vector search** (ChromaDB): computes cosine similarity between the query embedding and all stored embeddings using approximate nearest-neighbour (ANN) search.
- **BM25** (rank_bm25): computes a term-frequency / inverse-document-frequency score for each document. Uses `BM25Okapi` which applies document-length normalisation.

**Stage 2: Reciprocal Rank Fusion**

The two ranked lists have incompatible score scales — BM25 scores are raw TF-IDF values; cosine similarities are in `[0, 1]`. Normalising and then weighting them requires careful tuning per corpus.

RRF avoids this entirely by working on *ranks*, not scores:

```
rrf_score(d) = Σ  1 / (k + rank_i(d))
             all lists i where d appears
```

`k = 60` is the standard smoothing constant from the original paper. A document appearing at rank 1 in one list and rank 3 in another gets a higher fused score than a document appearing at rank 2 in both — the formula naturally rewards documents that both retrievers agree on.

**Why 50 candidates then rerank to 5?**

The cross-encoder reranker (Phase 3b) is too slow to run over the full corpus, but highly accurate on a small set. Fetching 50 candidates from each source and fusing to ~50 unique docs gives the reranker a strong candidate set to work from.

## Developer Benefits

- **No score normalisation needed**: RRF is rank-based. You never need to tune a mixing weight between BM25 and cosine scores.
- **Exact term recall**: Function names, error codes, CLI flags, and API endpoints are retrieved precisely by BM25 even when the embedding model blurs them.
- **Semantic recall**: "how does the pipeline handle errors?" retrieves relevant documents even without the word "errors" appearing verbatim.
- **Degrades gracefully**: If one retriever returns no results (e.g. BM25 finds no matching terms), the other still functions. The fused list is the union, not the intersection.

## Alternatives

| Alternative | Trade-offs |
|---|---|
| **Pure vector search** | Simple, good semantic recall. Misses exact terms, acronyms, identifiers. Fine for prose-only corpora with no technical jargon. |
| **Pure BM25** | Fast, exact matching. No semantic understanding. Misses paraphrases and synonyms. Good baseline for keyword-heavy search. |
| **Weighted score fusion** | Combines scores directly: `α * vec_score + (1-α) * bm25_score`. Requires normalising incompatible score scales and tuning `α` per corpus. More fragile than RRF. |
| **SPLADE / learned sparse retrieval** | Sparse vectors learned to expand query/document terms. Better than BM25 on semantic matching. Requires training or using a pre-trained model; higher inference cost than BM25. |
| **ColBERT** | Late-interaction dense retrieval. Very high recall. Requires specialised indexing and larger storage. Overkill for most use cases; worth evaluating if RRF+reranker is still missing results. |

**When to skip hybrid**: If your corpus is purely conversational prose with no technical identifiers, pure vector search is simpler and usually sufficient. Add BM25 when you see "keyword miss" failures in your trace logs.

## Effectiveness

From the BEIR benchmark (Thakur et al., 2021), which covers 18 retrieval tasks:

- Hybrid search (BM25 + dense) consistently outperforms either alone, with typical NDCG@10 improvements of **+3–8%** over the best single retriever.
- On code search and technical documentation tasks, the improvement is larger (**+10–15%**) because exact identifier matching strongly benefits from BM25.
- RRF specifically: Cormack et al. (2009) showed RRF outperforms learned rank combination methods on TREC benchmarks, with the benefit of zero tuning.

**Caveats**:
- BM25 assumes whitespace tokenisation. For compound words, camelCase, and snake_case identifiers, a smarter tokeniser (e.g. splitting on case changes) would improve recall.
- RRF's k=60 is a reasonable default but can be tuned if you have labeled evaluation data.

## Usage in atomic-rag

### DataPacket contract

**Input**: `DataPacket` with `query` set. `expanded_queries` is used when Phase 2 ships; currently only `query` is used.

**Output**: `DataPacket` with `documents` populated (top_k Documents sorted by score descending) and one `TraceEntry` appended.

### Minimal example

```python
from atomic_rag.ingestion import CodeIngestor
from atomic_rag.models.ollama import OllamaEmbedder
from atomic_rag.retrieval import HybridRetriever
from atomic_rag.schema import DataPacket

# Index
docs = CodeIngestor().ingest_directory("src/")
retriever = HybridRetriever(embedder=OllamaEmbedder())
retriever.add_documents(docs)

# Query
packet = DataPacket(query="how does authentication work?")
result = retriever.retrieve(packet, top_k=5)

for doc in result.documents:
    print(f"[{doc.score:.3f}] {doc.source}  {doc.content[:80]}")
```

### Without reranking

```python
retriever = HybridRetriever(embedder=OllamaEmbedder())  # reranker=None by default
```

### With cross-encoder reranking

```python
from atomic_rag.retrieval import CrossEncoderReranker

retriever = HybridRetriever(
    embedder=OllamaEmbedder(),
    reranker=CrossEncoderReranker(),   # adds ~100–300ms per query
    candidate_k=50,
)
```

### Persistent vector store (survives restarts)

```python
from atomic_rag.retrieval import ChromaVectorStore

retriever = HybridRetriever(
    embedder=OllamaEmbedder(),
    vector_store=ChromaVectorStore(persist_path="./chroma_db"),
)
```
