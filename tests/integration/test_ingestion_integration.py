"""
Integration tests for the ingestion module — call real MarkItDown.

These are skipped by default. Run them with:
    pytest -m integration

Requires: pip install markitdown
"""

from pathlib import Path

import pytest

from atomic_rag.ingestion import MarkItDownIngestor

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.mark.integration
def test_markitdown_ingests_markdown_file():
    ing = MarkItDownIngestor()
    docs = ing.ingest(FIXTURES / "sample.md")
    assert len(docs) > 0
    assert all(d.content.strip() for d in docs)
    assert all(d.chunk_index is not None for d in docs)


@pytest.mark.integration
def test_markitdown_source_is_absolute_path():
    ing = MarkItDownIngestor()
    docs = ing.ingest(FIXTURES / "sample.md")
    assert all(Path(d.source).is_absolute() for d in docs)
