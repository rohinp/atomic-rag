"""
Markdown-aware text chunker.

Splits a Markdown string into Document chunks using a two-level strategy:
  1. Primary split on header boundaries (## / ###) — preserves semantic sections.
     Headers are kept with their content so retrieved chunks are self-contained.
  2. Secondary split on paragraph boundaries (\n\n) for sections that exceed
     max_chunk_chars — avoids oversized chunks without breaking mid-sentence.

Why headers first?
  MarkItDown preserves document structure as Markdown headers. Splitting on them
  means each chunk corresponds to a meaningful section (a slide, a table, a
  document section) rather than an arbitrary window of characters. This improves
  retrieval precision because the chunk boundary matches a semantic boundary.
"""

import re
from pathlib import Path

from atomic_rag.schema import Document

# Lookahead so the header line is kept at the start of each part, not discarded.
_HEADER_RE = re.compile(r"(?=^#{1,6}\s)", re.MULTILINE)


class MarkdownChunker:
    """
    Split a Markdown string into Document chunks.

    Args:
        max_chunk_chars: Maximum character length of a single chunk.
                         Sections longer than this are split on paragraph
                         boundaries. Default 1000 chars (~200 tokens).
    """

    def __init__(self, max_chunk_chars: int = 1000) -> None:
        if max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be positive, got {max_chunk_chars}")
        self.max_chunk_chars = max_chunk_chars

    def chunk(self, text: str, source: str) -> list[Document]:
        """
        Split markdown text into Documents.

        Args:
            text:   The markdown string to chunk (e.g. output of MarkItDown).
            source: Logical identifier for the origin document (file path, URL, etc.).
                    Copied verbatim into every Document.source.

        Returns:
            Ordered list of Documents with chunk_index set. Returns [] if text
            is empty or contains only whitespace.
        """
        if not text or not text.strip():
            return []

        sections = [s.strip() for s in _HEADER_RE.split(text) if s.strip()]
        chunks: list[Document] = []
        idx = 0

        for section in sections:
            if len(section) <= self.max_chunk_chars:
                chunks.append(Document(content=section, source=source, chunk_index=idx))
                idx += 1
            else:
                idx = self._chunk_by_paragraphs(section, source, chunks, idx)

        return chunks

    def _chunk_by_paragraphs(
        self, text: str, source: str, chunks: list[Document], start_idx: int
    ) -> int:
        """
        Greedily accumulate paragraphs into chunks up to max_chunk_chars.
        Returns the next available chunk_index.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current: list[str] = []
        current_len = 0
        idx = start_idx

        for para in paragraphs:
            # +2 for the '\n\n' separator we'd add between paragraphs
            addition = len(para) + (2 if current else 0)
            if current and current_len + addition > self.max_chunk_chars:
                chunks.append(
                    Document(content="\n\n".join(current), source=source, chunk_index=idx)
                )
                idx += 1
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += addition

        if current:
            chunks.append(
                Document(content="\n\n".join(current), source=source, chunk_index=idx)
            )
            idx += 1

        return idx
