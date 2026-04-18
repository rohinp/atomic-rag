# Example: Code Q&A

Uses atomic-rag to index a Python codebase and answer questions about it.
Currently dogfoods against the atomic-rag source itself.

This example evolves as library phases ship — each step below becomes
functional when the corresponding phase is implemented.

## Setup

**Option A — Ollama (local, no API key)**

```bash
# Install Ollama: https://ollama.com
ollama pull nomic-embed-text    # embedding model
ollama pull llama3.2:3b         # chat model (fast, good for dev)
# ollama pull llama3.1:8b       # better quality, use for production
```

**Option B — OpenAI**

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

Then edit `config.py` to switch the provider (instructions in the file).

## Steps

### Step 1 — Ingest (works now)

```bash
# Index the atomic-rag codebase
python examples/code_qa/ingest.py

# Index a different repo
python examples/code_qa/ingest.py path/to/your/project
```

### Step 2 — Query (Phase 3 required)

```bash
python examples/code_qa/query.py
```

## What each step will do

| Step | Requires | What it does |
|---|---|---|
| `ingest.py` | nothing | AST-chunks Python files, shows chunk preview |
| `query.py` (v2) | Phase 3 | Embeds query, searches vector store + BM25, reranks |
| `query.py` (v3) | Phase 4 | Compresses retrieved chunks before LLM call |
| `query.py` (v4) | Phase 2 | Expands query with HyDE before retrieval |
| `query.py` (v5) | Phase 5 | LLM answers with C-RAG self-correction |

## Embedding models for code

General-purpose embedding models work but underperform on code.
Ranked by code retrieval quality:

| Model | Provider | Notes |
|---|---|---|
| `nomic-embed-code` | Ollama (local) | Best local option for code |
| `nomic-embed-text` | Ollama (local) | Good general-purpose, fine for mixed corpora |
| `voyage-code-2` | Voyage AI (API) | Best overall, paid |
| `text-embedding-3-small` | OpenAI (API) | Decent, cost-effective |
