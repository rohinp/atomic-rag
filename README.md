# atomic-rag

> Note: WIP I'm still reviewing the generated code, though the exmaples work :-) .

A modular Python library of research-backed RAG building blocks. Each component solves one specific failure mode of retrieval-augmented generation and can be used independently or composed into a full pipeline.

The design goal is the opposite of LangChain: no magic, no hidden abstractions. Every module has a clear input/output contract (`DataPacket`), is independently testable, and can be swapped without touching anything else.

---

## Install

**Full local stack with Ollama:**

```bash
pip install "atomic-rag-lib[all]"
```

Then pull the required Ollama models:

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2:3b        # chat / reasoning
```

**Pick only what you need:**

```bash
pip install atomic-rag-lib                     # core only (DataPacket + schema)
pip install "atomic-rag-lib[ollama]"           # local models via Ollama
pip install "atomic-rag-lib[openai]"           # OpenAI API models
pip install "atomic-rag-lib[retrieval]"        # ChromaDB + BM25
pip install "atomic-rag-lib[markitdown]"       # PDF/PPTX/XLSX ingestion
pip install "atomic-rag-lib[reranker]"         # cross-encoder reranking
pip install "atomic-rag-lib[ragas]"            # Ragas evaluation metrics
pip install "atomic-rag-lib[all]"              # everything above
```

**For development (clone the repo first):**

```bash
git clone https://github.com/rohinp/atomic-rag
cd atomic-rag
pip install -e ".[all,dev]"
```

## Quick Start

**Ingest a PDF or Office document:**

```python
from atomic_rag.ingestion import MarkItDownIngestor

ingestor = MarkItDownIngestor()
docs = ingestor.ingest("reports/q4-2024.pdf")

for doc in docs:
    print(f"[{doc.chunk_index}] {doc.content[:80]}...")
```

**Ingest a Python codebase (AST-based chunking):**

```python
from atomic_rag.ingestion import CodeIngestor

ingestor = CodeIngestor()
docs = ingestor.ingest_directory("src/")  # walks recursively, ignores __pycache__ etc.

for doc in docs:
    print(f"[{doc.chunk_index}] {doc.metadata['type']:<8} {doc.metadata.get('name', '')}  ({doc.source})")
```

## Development

```bash
pytest                          # run all tests (integration tests excluded)
pytest -m integration           # run integration tests (requires real dependencies)
pytest tests/test_ingestion.py  # run a single test file
pytest tests/test_ingestion.py::TestMarkdownChunker::test_splits_on_h2_headers  # single test
pytest --cov=atomic_rag --cov-report=term-missing  # with coverage
```

---

## Architecture

All modules communicate through a single `DataPacket` object that accumulates state as it moves through the pipeline. Modules never mutate their input — they return a copy with their output fields populated.

```
DataPacket(query="...")
  -> [Phase 2] expanded_queries populated
  -> [Phase 3] documents populated (retrieved + reranked, with scores)
  -> [Phase 4] context populated (compressed string for the LLM)
  -> [Phase 5] answer populated
  -> [Eval]    eval_scores populated (faithfulness, answer_relevance, context_precision)
```

Each phase also appends a `TraceEntry` to `packet.trace` for observability.

### Phases

| Phase | Problem solved | Key technique | Status |
|---|---|---|---|
| 1 — Ingestion | Messy PDFs destroy table/header structure | Markdown-native parsing (MarkItDown) + AST-based code chunking | **done** |
| 3 — Retrieval | Vector search misses keywords and acronyms | Hybrid search (vector + BM25) + RRF + cross-encoder reranking | **done** |
| 4 — Context | LLMs ignore information buried mid-context | Sentence-level cosine filtering (SentenceCompressor) | **done** |
| 2 — Query | Vague queries miss the relevant documents | HyDE + multi-query expansion | **done** |
| 5 — Agent | Hallucinations when retrieved context is insufficient | Corrective RAG (C-RAG) with evaluator + fallback | **done** |
| Eval | No visibility into where the pipeline fails | Faithfulness + answer relevance + Ragas integration | **done** |

Phase 3 before Phase 2 is intentional — hybrid retrieval delivers the highest quality improvement per unit of work. Query intelligence (Phase 2) has diminishing returns until retrieval is solid.

### Tech Stack

| Layer | Library |
|---|---|
| Parsing | Microsoft MarkItDown (swap: Docling) |
| Vector store | ChromaDB (swap: Qdrant) |
| Keyword search | rank-bm25 |
| Reranking | sentence-transformers cross-encoders |
| LLM / Embedder | Ollama (swap: OpenAI, or any ChatModelBase) |
| Evaluation | Built-in scorers + optional Ragas integration |

---

## Docs

Start at [`docs/index.md`](https://github.com/rohinp/atomic-rag/blob/main/docs/index.md) — it has a guided reading order, a full table of contents, and a pipeline diagram.

Quick links:
- [DataPacket contract](https://github.com/rohinp/atomic-rag/blob/main/docs/concepts/data-packet.md)
- [Ingestion module](https://github.com/rohinp/atomic-rag/blob/main/docs/modules/ingestion.md)
- [Retrieval module](https://github.com/rohinp/atomic-rag/blob/main/docs/modules/retrieval.md)
- [Hybrid search technique](https://github.com/rohinp/atomic-rag/blob/main/docs/techniques/hybrid-search.md)
- [Cross-encoder reranking](https://github.com/rohinp/atomic-rag/blob/main/docs/techniques/cross-encoder-reranking.md)
- [Markdown-native parsing](https://github.com/rohinp/atomic-rag/blob/main/docs/techniques/markdown-native-parsing.md)
- [Context module](https://github.com/rohinp/atomic-rag/blob/main/docs/modules/context.md)
- [Context compression technique](https://github.com/rohinp/atomic-rag/blob/main/docs/techniques/context-compression.md)
- [Query module](https://github.com/rohinp/atomic-rag/blob/main/docs/modules/query.md)
- [HyDE technique](https://github.com/rohinp/atomic-rag/blob/main/docs/techniques/hyde.md)
- [Multi-query expansion technique](https://github.com/rohinp/atomic-rag/blob/main/docs/techniques/multi-query-expansion.md)
- [Agent module](https://github.com/rohinp/atomic-rag/blob/main/docs/modules/agent.md)
- [Corrective RAG technique](https://github.com/rohinp/atomic-rag/blob/main/docs/techniques/corrective-rag.md)
- [Evaluation module](https://github.com/rohinp/atomic-rag/blob/main/docs/modules/evaluation.md)
- [Swapping backends guide](https://github.com/rohinp/atomic-rag/blob/main/docs/guides/swapping-backends.md)

## Examples

- [`examples/code_qa/`](https://github.com/rohinp/atomic-rag/tree/main/examples/code_qa) — Q&A over a Python codebase using the full pipeline
- [`examples/novel_qa/`](https://github.com/rohinp/atomic-rag/tree/main/examples/novel_qa) — Q&A over *The War of the Worlds* (plain text, auto-downloaded)
- [`examples/book_qa/`](https://github.com/rohinp/atomic-rag/tree/main/examples/book_qa) — Q&A over *Dive into Deep Learning* PDF (auto-downloaded)
