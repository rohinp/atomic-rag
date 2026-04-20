# atomic-rag Documentation

A modular Python library of research-backed RAG building blocks. Each component
solves one specific retrieval failure mode and can be used independently or
composed into a full pipeline.

---

## Where to start

**New to the library?** Read these three pages in order:

1. [The DataPacket contract](concepts/data-packet.md) — the single object that flows through every phase; understanding this makes everything else click
2. [Ingestion module](modules/ingestion.md) — how documents get into the system
3. [Retrieval module](modules/retrieval.md) — how relevant chunks get back out

**Want to understand why a technique was chosen?** Go to [Techniques](#techniques).

**Want to swap a component?** Each module page has a "Swapping the backend" section.

**Building the code Q&A example?** See [`examples/code_qa/README.md`](../examples/code_qa/README.md).

---

## Concepts

Core ideas that cut across the whole library.

| Page | What it covers |
|---|---|
| [DataPacket](concepts/data-packet.md) | The inter-module contract — schema, immutability rule, helpers, serialisation |

---

## Modules

Reference for each pipeline phase: input/output contract, configuration, error handling, how to run tests.

| Page | Phase | Status |
|---|---|---|
| [Ingestion](modules/ingestion.md) | Phase 1 — parse files into Document chunks | done |
| [Retrieval](modules/retrieval.md) | Phase 3 — hybrid search + reranking | done |
| [Context](modules/context.md) | Phase 4 — compress retrieved chunks before the LLM | done |
| Query Intelligence *(coming)* | Phase 2 — HyDE + multi-query expansion | planned |
| Agent *(coming)* | Phase 5 — answer generation with C-RAG | planned |
| Evaluation *(coming)* | Faithfulness + context precision scoring | planned |

---

## Techniques

Each page covers one research-backed technique: problem statement, how it works, developer benefits, alternatives with honest trade-offs, and effectiveness from benchmarks.

| Page | Used in | What it solves |
|---|---|---|
| [Markdown-Native Parsing](techniques/markdown-native-parsing.md) | Phase 1 | PDFs and tables that destroy structure when extracted as plain text |
| [Hybrid Search](techniques/hybrid-search.md) | Phase 3 | Vector search misses exact identifiers; BM25 misses semantic meaning |
| [Cross-Encoder Reranking](techniques/cross-encoder-reranking.md) | Phase 3 | First-stage retrieval buries the most relevant document at rank 8–15 |
| HyDE *(coming)* | Phase 2 | Vague queries that don't overlap with document vocabulary |
| Multi-Query Expansion *(coming)* | Phase 2 | Single query misses relevant documents from different angles |
| [Context Compression](techniques/context-compression.md) | Phase 4 | "Lost in the Middle" — LLMs ignore information buried in long contexts |
| Corrective RAG *(coming)* | Phase 5 | Hallucinations when retrieved context is insufficient or irrelevant |

---

## Guides

Task-oriented walkthroughs.

| Page | What it covers |
|---|---|
| [Swapping backends](guides/swapping-backends.md) *(coming)* | Replace ChromaDB with Qdrant, MarkItDown with Docling, Ollama with OpenAI |

---

## Pipeline at a glance

```
files
  └─ Ingestor ──────────────────────────── list[Document]
                                               │
                                        (load into vector store + BM25)
                                               │
query ──── DataPacket(query="...") ────── HybridRetriever
                                               │
                                    DataPacket(documents=[...])  ← scored + ranked
                                               │
                                      ContextCompressor          ← Phase 4 (done)
                                               │
                                    DataPacket(context="...")
                                               │
                                        AgentRunner              ← Phase 5 (planned)
                                               │
                                    DataPacket(answer="...")
```

Every phase appends a `TraceEntry` to `packet.trace` for observability.
Feed `packet.trace` to Langfuse for full-pipeline distributed tracing.
