# Ingestion Module

**Location**: `atomic_rag/ingestion/`

Converts raw files (PDF, PPTX, XLSX, DOCX, HTML, plain text) into ordered lists of `Document` chunks ready for loading into a vector store.

## Components

| Class | Role |
|---|---|
| `IngestorBase` | Abstract interface — subclass this to add a new parser |
| `MarkdownChunker` | Splits a Markdown string into `Document` chunks at header and paragraph boundaries |
| `MarkItDownIngestor` | Concrete ingestor: runs MarkItDown → MarkdownChunker → `list[Document]` |

See [Markdown-Native Parsing](../techniques/markdown-native-parsing.md) for the research rationale behind this approach.

## Input / Output

**Input**: A file path (`str` or `pathlib.Path`).

**Output**: `list[Document]`, ordered by position in the source. Each Document:

```python
Document(
    id="<uuid>",
    content="## Revenue\nTotal revenue reached $4.2B...",  # includes section header
    source="/abs/path/to/report.pdf",                       # always absolute
    chunk_index=2,                                          # 0-based position
    score=0.0,                                              # set later by retriever
    metadata={},                                            # parser can add page numbers etc.
)
```

This output is not a `DataPacket` — ingestion runs once at index time, before any query exists. The Documents are loaded directly into the vector store.

## Minimal Working Example

```python
from atomic_rag.ingestion import MarkItDownIngestor

ingestor = MarkItDownIngestor()
docs = ingestor.ingest("reports/q4-2024.pdf")

print(f"Produced {len(docs)} chunks")
for doc in docs[:3]:
    print(f"  [{doc.chunk_index}] {doc.content[:60]}...")
```

## Configuration

```python
from atomic_rag.ingestion import MarkItDownIngestor, MarkdownChunker

# Smaller chunks for models with limited context windows
ingestor = MarkItDownIngestor(
    chunker=MarkdownChunker(max_chunk_chars=500)
)

# Larger chunks to preserve more context per retrieval hit
ingestor = MarkItDownIngestor(
    chunker=MarkdownChunker(max_chunk_chars=2000)
)
```

**Choosing `max_chunk_chars`**:
- `500` (~100 tokens): good for precise fact retrieval; more chunks, higher recall cost
- `1000` (~200 tokens): default; balances precision and context
- `2000` (~400 tokens): better for reasoning over long passages; fewer chunks, may dilute relevance scores

## Error Handling

| Situation | Behaviour |
|---|---|
| File does not exist | `FileNotFoundError` with the path |
| MarkItDown produces empty output | `ValueError: MarkItDown produced no content for: <path>` |
| `markitdown` package not installed | `ImportError` with `pip install markitdown` instruction |

## Swapping the Backend

Implement `IngestorBase` to use a different parser. The rest of the pipeline only depends on the `list[Document]` return type:

```python
from pathlib import Path
from atomic_rag.ingestion.base import IngestorBase
from atomic_rag.ingestion.chunker import MarkdownChunker
from atomic_rag.schema import Document

class MyCustomIngestor(IngestorBase):
    def __init__(self, chunker=None):
        self.chunker = chunker or MarkdownChunker()

    def ingest(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        markdown = my_parser(str(path))          # your parser here
        return self.chunker.chunk(markdown, source=str(path))
```

## Running Tests

```bash
# Offline unit tests (no markitdown required)
pytest tests/test_ingestion.py -v

# Integration tests (requires: pip install markitdown)
pytest -m integration tests/integration/test_ingestion_integration.py -v
```
