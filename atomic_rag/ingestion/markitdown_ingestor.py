"""
Ingestor implementation backed by Microsoft MarkItDown.

MarkItDown converts PDFs, Word docs, PowerPoint, Excel, HTML, and plain text
into structured Markdown. This ingestor wraps it, feeds the output to
MarkdownChunker, and returns a list of Documents.

The `_converter` parameter exists solely for testing — it lets tests inject a
mock instead of calling MarkItDown for real.

Supported file types (requires markitdown[all] installed):
  .pdf, .docx, .pptx, .xlsx, .html, .htm, .txt, .md, .csv, .json, .xml
"""

from pathlib import Path
from typing import Any

from atomic_rag.ingestion.base import IngestorBase
from atomic_rag.ingestion.chunker import MarkdownChunker
from atomic_rag.schema import Document


class MarkItDownIngestor(IngestorBase):
    """
    Parse a file to Markdown via MarkItDown, then chunk it into Documents.

    Args:
        chunker:    Chunking strategy. Defaults to MarkdownChunker().
        _converter: Optional MarkItDown instance. Injected in tests to avoid
                    hitting the real library. In production leave this as None.
    """

    def __init__(
        self,
        chunker: MarkdownChunker | None = None,
        _converter: Any = None,
    ) -> None:
        self.chunker = chunker or MarkdownChunker()
        self._converter = _converter  # resolved lazily if None

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            from markitdown import MarkItDown  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "markitdown is required for MarkItDownIngestor. "
                "Install it with: pip install markitdown"
            ) from e
        return MarkItDown()

    def ingest(self, file_path: str | Path) -> list[Document]:
        """
        Convert file_path to Markdown and return it as chunked Documents.

        Args:
            file_path: Path to the file. Must exist and be readable.

        Returns:
            Ordered list of Documents. chunk_index is 0-based position within
            the source file. source is set to the resolved absolute path string.

        Raises:
            FileNotFoundError: if file_path does not exist.
            ValueError: if MarkItDown produces empty output.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        converter = self._get_converter()
        result = converter.convert(str(path))
        markdown = result.text_content

        if not markdown or not markdown.strip():
            raise ValueError(f"MarkItDown produced no content for: {path}")

        return self.chunker.chunk(markdown, source=str(path))
