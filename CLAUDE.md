# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**atomic-rag** is a modular RAG (Retrieval-Augmented Generation) framework designed as a "best-of-breed building block library." The core philosophy is to address the "Seven Failure Points" of RAG through swappable, independently debuggable modules rather than monolithic pipelines (like LangChain).

## Architecture

The system is organized into 5 sequential phases, each solving a specific RAG failure point:

| Phase | Failure Point Addressed | Key Technique | Primary Library |
|---|---|---|---|
| 1 - Data Ingestion | Messy PDFs/tables ("garbage in") | Markdown-native parsing | Microsoft MarkItDown |
| 2 - Query Intelligence | Vague queries miss gold documents | HyDE + Multi-Query Expansion | DSPy |
| 3 - Precision Retrieval | Vector search misses keywords/acronyms | Hybrid Search (Vector + BM25) + Reranking | ChromaDB + Sentence-Transformers |
| 4 - Context Engineering | "Lost in the Middle" problem | Dynamic Context Compression | — |
| 5 - Agentic Reasoning | Hallucinations / incomplete answers | Corrective RAG (C-RAG) with evaluator agent | Agno |

**Evaluation layer**: Ragas (faithfulness + context precision metrics) + Langfuse (full-trace observability).

## Design Principles

- **Modularity over monoliths**: Every phase is a swappable component. Changing a reranker should not break the agent.
- **Standardized inter-module interface**: Modules pass data via a standardized JSON schema (to be defined). This is the key contract — future code must respect this boundary.
- **Research-backed implementations**: Prefer well-researched techniques (HyDE, C-RAG, BGE-Reranker) over convenience wrappers.
- **Granular observability**: Each module should be independently observable so failures can be pinpointed to a specific phase.

## Testing Requirements

Every piece of code written in this project must have tests. No exceptions.

- **Framework**: `pytest` with `pytest-cov` for coverage
- **Run all tests**: `pytest`
- **Run a single test file**: `pytest tests/test_<module>.py`
- **Run a single test**: `pytest tests/test_<module>.py::test_function_name`
- **Coverage report**: `pytest --cov=atomic_rag --cov-report=term-missing`

**Testing conventions:**
- Each module in `atomic_rag/` has a corresponding `tests/test_<module>.py`
- Use `pytest.fixture` for shared setup (sample documents, mock embeddings, etc.)
- Use `unittest.mock` or `pytest-mock` to stub external API calls (LLMs, embedding models) — tests must be runnable offline
- Integration tests that hit real models go in `tests/integration/` and are skipped by default (`@pytest.mark.integration`)
- Each phase module must expose a testable pure-function core — side effects (network, disk) are injected dependencies

## Documentation (Wiki)

User-facing docs live in `docs/`. Structure:

```
docs/
  index.md                  # what atomic-rag is, when to use it, quickstart
  concepts/
    data-packet.md          # the inter-module contract explained
  techniques/
    hyde.md
    multi-query-expansion.md
    hybrid-search.md
    cross-encoder-reranking.md
    context-compression.md
    corrective-rag.md
  modules/
    ingestion.md
    retrieval.md
    context.md
    query.md
    agent.md
    evaluation.md
  guides/
    swapping-backends.md    # how to replace e.g. ChromaDB with Qdrant
```

**`docs/index.md` is the entry point.** Keep it updated whenever a new module or technique page is added: add a row to the relevant table and update its status (`done` / `planned`). The pipeline diagram at the bottom should also reflect new phases as they ship.

**Every technique page in `docs/techniques/` must follow this exact structure:**

1. **Problem Statement** — what failure mode this solves, with a concrete example of what goes wrong without it
2. **How It Works** — a plain-English explanation of the mechanism, followed by a brief description of the math/algorithm where relevant
3. **Developer Benefits** — what this buys you in practice (latency, recall, cost, debuggability)
4. **Alternatives** — other approaches to the same problem, with honest trade-offs (when would you pick the alternative instead?)
5. **Effectiveness** — what the research says: benchmarks, datasets, typical improvement ranges, and caveats
6. **Usage in atomic-rag** — the input/output `DataPacket` contract for this technique and a minimal code example

**Every module page in `docs/modules/` must include:**
- Input/output `DataPacket` contract
- Minimal working example (runnable, no hidden setup)
- How to swap the backend implementation

## README Maintenance

**Keep `README.md` updated with every change.** Specifically:

- When a phase ships, change its status from `planned` to `done` in the phases table.
- When a new module is added, add a link to its docs page in the Docs section.
- When a new technique wiki page is written, add it to the Docs section.
- The Quick Start example must always reflect working, currently-implemented code.
- Never add marketing language, emojis, or "Why This Works" sections to the README.

## Build & Dev Setup

```bash
pip install -e ".[dev]"   # installs package + dev dependencies (pytest, pytest-cov, etc.)
pytest                     # run all tests
```

## Status

This project is in the **design/planning phase**. No implementation code exists yet. The immediate next step is defining the standardized JSON interface for inter-module data passing before any phase is implemented.

**Build order** (highest ROI first):
1. Define inter-module data contract (`DataPacket` schema)
2. Phase 1: Ingestion/parsing
3. Phase 3: Hybrid retrieval + reranking
4. Phase 4: Context compression
5. Phase 2: Query intelligence (HyDE, multi-query)
6. Phase 5: Agentic C-RAG loop
7. Evaluation layer woven into each phase as it ships

## Tech Stack (Planned)

- **Language**: Python (ML/AI ecosystem)
- **Parsing**: Microsoft MarkItDown
- **Orchestration**: DSPy / Agno
- **Vector Store**: ChromaDB or Qdrant
- **Reranking**: Mixedbread.ai or BGE-Reranker
- **Evaluation**: Ragas
- **Monitoring**: Langfuse
