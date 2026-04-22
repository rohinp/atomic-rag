# book_qa — Q&A over Dive into Deep Learning

Ask natural-language questions about **Dive into Deep Learning** (d2l.ai)
using the full atomic-rag pipeline. The PDF (~45 MB) is downloaded automatically
on first run and cached locally.

## Setup

```bash
pip install -e ".[dev]"
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Usage

```bash
# Ask a single question (PDF downloads and indexes automatically on first run)
python examples/book_qa/query.py "what is backpropagation?"

# Query expansion strategies (improve recall for vague questions)
python examples/book_qa/query.py --hyde "how does attention mechanism work?"
python examples/book_qa/query.py --multi-query "explain gradient descent"

# Debug mode — show retrieved chunks, context, and raw evaluator score
python examples/book_qa/query.py --verbose "what is a transformer?"

# Show retrieved context without generating an answer
python examples/book_qa/query.py --no-answer "what is dropout?"

# Interactive mode
python examples/book_qa/query.py
```

Preview how the book is chunked (optional):

```bash
python examples/book_qa/ingest.py
python examples/book_qa/ingest.py --redownload        # force fresh download
python examples/book_qa/ingest.py --chunk-size 2000   # larger chunks
```

## Pipeline

| Phase | Component | Detail |
|---|---|---|
| 1 — Ingestion | `MarkItDownIngestor` | PDF → Markdown via MarkItDown; chunks into ~1500-char sections |
| 2 — Query expansion | `HyDEExpander` / `MultiQueryExpander` | Optional; improves recall for technical terminology |
| 3 — Retrieval | `HybridRetriever` | Vector (nomic-embed-text) + BM25 fused via RRF |
| 4 — Compression | `SentenceCompressor` | Drops sentences with low semantic similarity to the query |
| 5 — Answer | `AgentRunner` (C-RAG) | Evaluates context quality before generating; falls back if context is insufficient |

## Notes

- **First run is slow** — PDF parsing + embedding ~1000 chunks takes 2–5 minutes.
  Subsequent runs reuse the cached PDF and re-index in the same session.
- Chunks are 1500 chars (~300 tokens) by default. Technical content like
  derivations and code blocks often spans multiple paragraphs, so larger chunks
  than prose examples give better coherence. Adjust with `--chunk-size`.
- `threshold=0.2` is set for `llama3.2:3b`. Raise to `0.5` when using a
  capable API model (edit `query.py`).
- For OpenAI models, edit `config.py`.
- The `data/` directory is git-ignored; the PDF is re-downloaded if missing.
