# Example: Code Q&A

Uses atomic-rag to index a Python codebase and answer questions about it.
Dogfoods against the atomic-rag source itself by default.

Full pipeline implemented:
- **Phase 1** — AST-based chunking of Python files (`CodeIngestor`)
- **Phase 2** — Query expansion via HyDE or multi-query (`--hyde`, `--multi-query`)
- **Phase 3** — Hybrid retrieval: vector search + BM25 + RRF + optional reranking
- **Phase 4** — Context compression: sentence-level cosine filtering
- **Phase 5** — C-RAG answer generation with context quality gate

## Setup

**Option A — Ollama (local, no API key)**

```bash
# 1. Install dependencies (from the repo root)
pip install -e ".[all,dev]"

# 2. Install Ollama: https://ollama.com, then pull models
ollama pull nomic-embed-text    # embedding model
ollama pull llama3.2:3b         # chat model
```

**Option B — OpenAI**

```bash
pip install -e ".[dev,retrieval,reranker,markitdown,openai]"
export OPENAI_API_KEY=sk-...
```

Then edit `config.py` to uncomment the OpenAI block.

## Running

### Step 1 — Inspect the index

```bash
# Preview chunks from the atomic-rag source
python examples/code_qa/ingest.py

# Preview chunks from a different repo
python examples/code_qa/ingest.py path/to/your/project
```

### Step 2 — Ask questions

```bash
# Basic: retrieve + compress + answer
python examples/code_qa/query.py "how does DataPacket work?"

# With HyDE query expansion (better for short/vague questions)
python examples/code_qa/query.py --hyde "what does the chunker do?"

# With multi-query expansion (better for vocabulary variance)
python examples/code_qa/query.py --multi-query "how does retrieval work?"

# Show only the compressed context — skip LLM answer
python examples/code_qa/query.py --no-answer "what is SentenceCompressor?"

# Interactive mode
python examples/code_qa/query.py
```

Each run rebuilds the index in-memory (fast for a codebase this size). The output shows timing and stats for each phase:

```
Building index... 147 chunks indexed in 3.2s

Query    : how does DataPacket work?
Retrieval: 84ms  (queries=1, vector=50, bm25=12)
Compress : 210ms  (18 → 7 sentences, 61% removed)
Agent    : 1430ms  (eval_score=0.82, fallback=False)

--- Answer ---

DataPacket is a Pydantic model that flows through every phase of the pipeline...
```

## Embedding models for code

General-purpose embedding models work but underperform on code.
Ranked by code retrieval quality:

| Model | Provider | Notes |
|---|---|---|
| `nomic-embed-code` | Ollama (local) | Best local option for code |
| `nomic-embed-text` | Ollama (local) | Good general-purpose, fine for mixed corpora |
| `voyage-code-2` | Voyage AI (API) | Best overall, paid |
| `text-embedding-3-small` | OpenAI (API) | Decent, cost-effective |

To switch, edit `config.py`.
