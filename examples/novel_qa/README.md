# novel_qa — Q&A over The War of the Worlds

Ask natural-language questions about **The War of the Worlds** by H. G. Wells
(Project Gutenberg, public domain) using the full atomic-rag pipeline.

## Setup

```bash
pip install -e ".[dev]"
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Usage

```bash
# Ask a single question (novel downloads automatically on first run)
python examples/novel_qa/query.py "what are the Martians like?"

# Query expansion strategies
python examples/novel_qa/query.py --hyde "how do humans fight back?"
python examples/novel_qa/query.py --multi-query "what happens in the end?"

# Debug mode — show retrieved chunks, context, and raw evaluator score
python examples/novel_qa/query.py --verbose "describe the heat-ray"

# Show retrieved context without generating an answer
python examples/novel_qa/query.py --no-answer "who is the narrator?"

# Interactive mode
python examples/novel_qa/query.py
```

Preview how the novel is chunked (optional):

```bash
python examples/novel_qa/ingest.py
python examples/novel_qa/ingest.py --redownload  # force fresh download
```

## Pipeline

| Phase | Component | Detail |
|---|---|---|
| 1 — Ingestion | `MarkItDownIngestor` | Downloads novel once; chunks prose into ~800-char paragraphs |
| 2 — Query expansion | `HyDEExpander` / `MultiQueryExpander` | Optional; improves recall for vague questions |
| 3 — Retrieval | `HybridRetriever` | Vector (nomic-embed-text) + BM25 fused via RRF |
| 4 — Compression | `SentenceCompressor` | Drops sentences with low semantic similarity to the query |
| 5 — Answer | `AgentRunner` (C-RAG) | Evaluates context quality before generating; falls back if context is insufficient |

## Sample output

```
Already downloaded: examples/novel_qa/data/war_of_the_worlds.txt
Building index... 312 chunks indexed in 18.4s

Query    : what are the Martians like?
Retrieval: 210ms  (queries=1, vector=8, bm25=5)
Compress : 85ms   (18 → 11 sentences, 39% removed)
Agent    : 3420ms (eval_score=0.82, threshold=0.2, fallback=False)

--- Answer ---

The Martians are described as vast, bear-like creatures with a large,
rounded body and sixteen slender tentacles arranged around a central mouth.
They have no digestive system — instead they absorb nutrients directly from
the blood of living creatures. Their immense intelligence is paired with
physical frailty on Earth due to the higher gravity.

--- End answer ---
```

## Notes

- The novel text is downloaded once to `data/war_of_the_worlds.txt` and reused on subsequent runs.
- The `data/` directory is git-ignored.
- `threshold=0.2` is set for `llama3.2:3b`. Raise to `0.5` when using a capable API model (edit `query.py`).
- For OpenAI models, edit `config.py`.
