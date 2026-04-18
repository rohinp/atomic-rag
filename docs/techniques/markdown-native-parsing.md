# Markdown-Native Parsing

## Problem Statement

Most RAG pipelines extract text from PDFs by reading raw bytes. This works for body paragraphs, but silently destroys structure:

- A table becomes a stream of whitespace-separated numbers with no row/column relationships.
- A PowerPoint slide becomes a flat blob of bullet text with no slide boundaries.
- Headers, section titles, and captions — which tell the LLM *what* a chunk is about — are lost or garbled.

The downstream effect is severe: a chunk containing quarterly revenue figures with no table headers is effectively uninterpretable. The retriever may find it; the LLM cannot use it. This is Failure Point 1 — "garbage in, garbage out."

**Concrete example:**

A PDF table like:

| Product | Q3 Revenue | Q4 Revenue |
|---------|-----------|-----------|
| Cloud   | $1.2B     | $1.6B     |

becomes raw text: `Product Q3 Revenue Q4 Revenue Cloud 1.2B 1.6B` — indistinguishable from prose.

## How It Works

[Microsoft MarkItDown](https://github.com/microsoft/markitdown) is a library that converts documents to Markdown rather than plain text. It uses format-specific parsers:

- **PDF**: Extracts text with layout analysis, preserves table structure as Markdown `|` tables.
- **PowerPoint (.pptx)**: Treats each slide as a section, slide titles become `##` headers.
- **Excel (.xlsx)**: Converts each sheet's data into a Markdown table.
- **Word (.docx)**: Maps heading styles to `#` / `##` / `###` headers.
- **HTML**: Converts semantic tags (`<h1>`, `<table>`, `<li>`) to their Markdown equivalents.

The output is a single Markdown string. `MarkdownChunker` then splits this on header boundaries, keeping each header with its content so every chunk is self-contained.

**Why headers matter for retrieval:**

A chunk that reads `## Revenue\nTotal revenue reached $4.2B` gives the embedding model the semantic context (`Revenue`) alongside the fact. A chunk that reads only `Total revenue reached $4.2B` is ambiguous — the embedding must guess the topic from the fact alone. Retrieval recall improves when the query "Q4 revenue figures" can match the header as well as the content.

## Developer Benefits

- **No pre/post-processing for tables**: MarkItDown handles the hard part. You get valid Markdown tables without writing a single regex.
- **Consistent chunking**: Because all formats output Markdown, `MarkdownChunker` works identically regardless of source format — one chunker for everything.
- **Debuggable intermediate**: The raw Markdown output is human-readable. When a chunk is wrong, you can inspect `result.text_content` directly rather than debugging binary byte offsets.
- **Header-as-context**: Retrieved chunks include their section header, so the LLM always sees "what section does this come from" without needing to retrieve parent documents separately.

## Alternatives

| Alternative | Trade-offs |
|---|---|
| **PyMuPDF / pdfminer** (raw text) | Fast and lightweight. Tables are destroyed. Good if documents are prose-only. |
| **Unstructured.io** | Richer element detection (figures, titles, list items as typed objects). More complex setup, heavier dependency, commercial for advanced features. Use if you need image extraction or fine-grained element classification. |
| **AWS Textract / Google Document AI** | Excellent table and form extraction. Requires cloud credentials, adds latency, costs money per page. Use for scanned PDFs (image-only) where MarkItDown falls short. |
| **LlamaParse** | Strong PDF parsing, Markdown output. API-based (not local), paid service. Good alternative if MarkItDown struggles with complex PDFs in production. |
| **Docling (IBM)** | Local, high-quality PDF-to-Markdown, strong table handling. Heavier install (`torch` dependency). Use as a drop-in MarkItDown replacement if quality is insufficient. |

**When to skip MarkItDown**: If your source documents are already plain text or clean HTML, the overhead of MarkItDown is unnecessary. Use `MarkdownChunker` directly on the raw text.

## Effectiveness

There is no universal benchmark for parsing quality, but the mechanism is well-understood:

- **Table structure preservation** is the primary win. Studies on financial document QA (e.g. FinQA, TAT-QA) show that models answering table-based questions score 20–40% higher when tables are presented as structured Markdown vs. flattened text.
- **Header-aware chunking** improves retrieval recall on multi-section documents because the embedding of "## Revenue\nTotal revenue..." is closer to "Q4 revenue figures" than the embedding of "Total revenue..." alone.
- **Caveats**: MarkItDown struggles with (1) scanned PDFs (no OCR), (2) complex multi-column layouts, (3) PDFs where text is encoded as images. For these, use AWS Textract or Google Document AI instead.

## Usage in atomic-rag

### Input / Output

`MarkItDownIngestor.ingest()` takes a file path and returns `list[Document]`.

Each `Document` has:
- `content`: a self-contained Markdown chunk (includes its section header)
- `source`: absolute path to the source file
- `chunk_index`: 0-based position within the source
- `score`: `0.0` (set later by the retriever)

No `DataPacket` is involved at this stage — ingestion happens before a query exists. The returned `list[Document]` is what you load into your vector store.

### Minimal Example

```python
from atomic_rag.ingestion import MarkItDownIngestor, MarkdownChunker

# Default settings
ingestor = MarkItDownIngestor()
docs = ingestor.ingest("reports/q4-2024.pdf")

for doc in docs:
    print(f"[{doc.chunk_index}] {doc.content[:80]}...")

# Custom chunk size (e.g. for models with small context windows)
ingestor = MarkItDownIngestor(chunker=MarkdownChunker(max_chunk_chars=500))
docs = ingestor.ingest("slides/product-overview.pptx")
```

### Swapping the Parser

To replace MarkItDown with a different parser (e.g. Docling), subclass `IngestorBase`:

```python
from pathlib import Path
from atomic_rag.ingestion.base import IngestorBase
from atomic_rag.ingestion.chunker import MarkdownChunker
from atomic_rag.schema import Document

class DoclingIngestor(IngestorBase):
    def __init__(self, chunker=None):
        self.chunker = chunker or MarkdownChunker()

    def ingest(self, file_path: str | Path) -> list[Document]:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        markdown = result.document.export_to_markdown()
        return self.chunker.chunk(markdown, source=str(Path(file_path).resolve()))
```

The rest of the pipeline is unaffected — it only sees `list[Document]`.
