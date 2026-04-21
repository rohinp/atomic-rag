# Retrieval Module

**Location**: `atomic_rag/retrieval/`

Implements Phase 3: takes a query and returns the most relevant `Document` chunks from an indexed corpus. Uses hybrid search (vector + BM25) fused with RRF, optionally followed by cross-encoder reranking.

## Components

| Class / Function | Role |
|---|---|
| `RetrieverBase` | Abstract interface — subclass to add a new retrieval strategy |
| `ChromaVectorStore` | Stores and searches document embeddings via ChromaDB |
| `BM25Retriever` | Keyword search via rank_bm25 |
| `reciprocal_rank_fusion()` | Pure function — merges ranked lists without score normalisation |
| `CrossEncoderReranker` | Re-scores candidates with a cross-encoder model |
| `HybridRetriever` | Orchestrates all of the above; main entry point |

See:
- [Hybrid Search](../techniques/hybrid-search.md) — why vector + BM25 + RRF, research backing, alternatives
- [Cross-Encoder Reranking](../techniques/cross-encoder-reranking.md) — why reranking matters, latency/quality trade-offs

## DataPacket Contract

**Input**: `DataPacket` with `query` set. If `expanded_queries` is populated (Phase 2 ran), vector search runs once per expanded query and all result lists are fused via RRF before the BM25 pass.

**Output**: New `DataPacket` with:
- `documents` — top_k `Document` objects sorted by `score` descending
- `trace` — one `TraceEntry` appended (`phase="retrieval"`)

The retriever never mutates the input packet.

## Install

```bash
pip install chromadb rank-bm25            # retrieval dependencies
pip install sentence-transformers          # reranker (optional)
```

Or via extras:
```bash
pip install "atomic-rag[retrieval,reranker]"
```

## Minimal Working Example

```python
from atomic_rag.ingestion import CodeIngestor
from atomic_rag.models.ollama import OllamaEmbedder
from atomic_rag.retrieval import HybridRetriever
from atomic_rag.schema import DataPacket

# 1. Index documents (once)
docs = CodeIngestor().ingest_directory("src/")
retriever = HybridRetriever(embedder=OllamaEmbedder())
retriever.add_documents(docs)

# 2. Retrieve (per query)
packet = DataPacket(query="how does the chunker split markdown?")
result = retriever.retrieve(packet, top_k=5)

for doc in result.documents:
    print(f"[{doc.score:.3f}] {doc.source}:{doc.metadata.get('start_line', '?')}")
    print(f"  {doc.content[:120]}")
```

## Configuration

### Without reranker (faster, lower precision)

```python
retriever = HybridRetriever(embedder=OllamaEmbedder())
```

### With reranker (recommended for production)

```python
from atomic_rag.retrieval import CrossEncoderReranker

retriever = HybridRetriever(
    embedder=OllamaEmbedder(),
    reranker=CrossEncoderReranker(),   # cross-encoder/ms-marco-MiniLM-L-6-v2
    candidate_k=50,                    # candidates per source before reranking
)
```

### Persistent index (survives restarts)

```python
from atomic_rag.retrieval import ChromaVectorStore

retriever = HybridRetriever(
    embedder=OllamaEmbedder(),
    vector_store=ChromaVectorStore(persist_path="./chroma_db"),
)
# Only call add_documents() once — subsequent runs load from disk
```

## Choosing candidate_k

`candidate_k` controls how many results each source (vector, BM25) retrieves before fusion and reranking.

| candidate_k | Best for |
|---|---|
| 20 | Small corpus (< 500 docs), fast queries |
| 50 | Default — good balance for most repos |
| 100 | Large corpus where the relevant doc may rank 40–80 in first stage |

## Observability

Every `retrieve()` call appends a `TraceEntry` with:

```python
{
    "phase": "retrieval",
    "duration_ms": 142.3,
    "details": {
        "queries_used": 3,      # 1 = no expansion; N = expanded_queries count
        "vector_hits": 144,     # total docs across all vector searches
        "bm25_hits": 23,        # docs returned by BM25 (fewer if query terms are rare)
        "fused_candidates": 51, # unique docs after RRF
        "top_k": 5,
        "reranked": True,
    }
}
```

Feed `packet.trace` to Langfuse for full-pipeline tracing.

## Swapping the Backend

Implement `RetrieverBase` to replace the whole retrieval strategy:

```python
from atomic_rag.retrieval.base import RetrieverBase
from atomic_rag.schema import DataPacket, Document

class MyCustomRetriever(RetrieverBase):
    def add_documents(self, documents: list[Document]) -> None:
        ...  # index into your store

    def retrieve(self, packet: DataPacket, top_k: int = 5) -> DataPacket:
        ...  # search and return updated packet
```

## Running Tests

```bash
pytest tests/test_retrieval.py -v

# Integration tests (requires Ollama + chromadb + rank-bm25 installed)
pytest -m integration tests/integration/test_retrieval_integration.py -v
```
