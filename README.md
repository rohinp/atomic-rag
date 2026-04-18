# atomic-rag

A modular Python library of research-backed RAG building blocks. Each component solves one specific failure mode of retrieval-augmented generation and can be used independently or composed into a full pipeline.

The design goal is the opposite of LangChain: no magic, no hidden abstractions. Every module has a clear input/output contract (`DataPacket`), is independently testable, and can be swapped without touching anything else.

---

## Install

```bash
pip install -e ".[dev]"              # development install with test dependencies
pip install markitdown               # for PDF/PPTX/XLSX ingestion
pip install ".[ollama]"              # for local models via Ollama
pip install ".[openai]"              # for OpenAI API models
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
  -> [Eval]    eval_scores populated (faithfulness, context_precision)
```

Each phase also appends a `TraceEntry` to `packet.trace` for observability.

### Phases

| Phase | Problem solved | Key technique | Status |
|---|---|---|---|
| 1 — Ingestion | Messy PDFs destroy table/header structure | Markdown-native parsing (MarkItDown) + AST-based code chunking | **done** |
| 3 — Retrieval | Vector search misses keywords and acronyms | Hybrid search (vector + BM25) + cross-encoder reranking | planned |
| 4 — Context | LLMs ignore information buried mid-context | Dynamic context compression | planned |
| 2 — Query | Vague queries miss the relevant documents | HyDE + multi-query expansion | planned |
| 5 — Agent | Hallucinations when retrieved context is insufficient | Corrective RAG (C-RAG) with evaluator + fallback | planned |
| Eval | No visibility into where the pipeline fails | Ragas (faithfulness, context precision) + Langfuse traces | planned |

Phase 3 before Phase 2 is intentional — hybrid retrieval delivers the highest quality improvement per unit of work. Query intelligence (Phase 2) has diminishing returns until retrieval is solid.

### Tech Stack

| Layer | Library |
|---|---|
| Parsing | Microsoft MarkItDown |
| Vector store | ChromaDB / Qdrant |
| Reranking | BGE-Reranker / Mixedbread.ai |
| Orchestration | DSPy / Agno |
| Evaluation | Ragas |
| Observability | Langfuse |

---

## Docs

Full documentation lives in [`docs/`](docs/):

- [`docs/concepts/data-packet.md`](docs/concepts/data-packet.md) — the inter-module contract
- [`docs/modules/ingestion.md`](docs/modules/ingestion.md) — ingestion module reference
- [`docs/techniques/markdown-native-parsing.md`](docs/techniques/markdown-native-parsing.md) — why and how markdown-native parsing works

## Examples

- [`examples/code_qa/`](examples/code_qa/) — indexes a Python codebase and answers questions about it (grows as phases ship)
