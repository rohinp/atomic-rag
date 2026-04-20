# Context Module

**Location**: `atomic_rag/context/`

Implements Phase 4: compresses retrieved documents into a clean context string for the LLM by dropping sentences with low semantic similarity to the query.

## Components

| Class | Role |
|---|---|
| `CompressorBase` | Abstract interface — subclass to add a new compression strategy |
| `SentenceCompressor` | Filters prose by sentence-level cosine similarity; keeps code chunks whole |
| `split_sentences()` | Utility — regex sentence splitter, no NLTK dependency |
| `cosine_similarity()` | Utility — pure vector dot-product similarity |

See [Context Compression](../techniques/context-compression.md) for the research rationale, alternatives, and effectiveness data.

## DataPacket Contract

**Input**: `DataPacket` with `query` and `documents` set (Phase 3 output).

**Output**: New `DataPacket` with:
- `context` — compressed string with source labels, ready for the LLM
- `trace` — one `TraceEntry` appended (`phase="context"`)

Input packet is never mutated.

## Minimal Working Example

```python
from atomic_rag.context import SentenceCompressor
from atomic_rag.models.ollama import OllamaEmbedder

# Use the same embedder as retrieval — keeps vector space consistent
compressor = SentenceCompressor(embedder=OllamaEmbedder(), threshold=0.5)

# packet has query + documents from Phase 3
result = compressor.compress(packet)

print(result.context)
# [Source: src/auth/service.py  L42–61]
# def authenticate(token: str) -> User: ...
#
# ---
#
# [Source: reports/q4.pdf  chunk 3]
# Cloud revenue grew 34% to $1.6B in Q4.

print(result.trace[-1].details)
# {'documents': 5, 'sentences_before': 28, 'sentences_after': 11,
#  'reduction_pct': 60.7, 'threshold': 0.5}
```

## Configuration

| Parameter | Default | Effect |
|---|---|---|
| `threshold` | `0.5` | Cosine similarity cutoff. Raise to filter more aggressively; lower for permissive contexts. |
| `min_sentences` | `1` | Minimum segments kept per document regardless of score. Prevents a document from disappearing entirely. |

### Tuning threshold

Start at `0.5` and inspect `reduction_pct` in traces:
- If `reduction_pct` < 20%: threshold may be too low — chunks already tight, or try raising it
- If `reduction_pct` > 70%: check that relevant sentences aren't being dropped; lower the threshold or raise `min_sentences`

## Context Format

The output string uses source labels and `---` separators:

```
[Source: /abs/path/to/file.py  L10–25]
<compressed content>

---

[Source: /abs/path/to/report.pdf  chunk 2]
<compressed content>
```

Source labels include line numbers for code chunks and chunk index for document chunks. These let the LLM cite sources accurately.

## Behaviour by Content Type

| Content type | Detected by | Split strategy |
|---|---|---|
| Prose (PDF, docs) | No `language` in metadata | Split on sentence boundaries |
| Code (Python, etc.) | `metadata["language"]` set | Kept as a single unit |

Code chunks are never split — a partial function body is syntactically invalid and semantically useless to the LLM.

## Running Tests

```bash
pytest tests/test_context.py -v
```
