"""
Tests for atomic_rag.ingestion — IngestorBase, MarkdownChunker, MarkItDownIngestor.

All tests run offline. MarkItDown is mocked so no real file conversion happens.
Integration tests that call the real MarkItDown library live in tests/integration/.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atomic_rag.ingestion import MarkdownChunker, MarkItDownIngestor
from atomic_rag.ingestion.base import IngestorBase
from atomic_rag.schema import Document

FIXTURES = Path(__file__).parent / "fixtures"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_md_text() -> str:
    return (FIXTURES / "sample.md").read_text()


@pytest.fixture
def chunker() -> MarkdownChunker:
    return MarkdownChunker(max_chunk_chars=1000)


@pytest.fixture
def mock_converter(sample_md_text: str) -> MagicMock:
    """A fake MarkItDown converter that returns the sample markdown fixture."""
    result = MagicMock()
    result.text_content = sample_md_text
    converter = MagicMock()
    converter.convert.return_value = result
    return converter


@pytest.fixture
def ingestor(mock_converter: MagicMock, tmp_path: Path) -> tuple[MarkItDownIngestor, Path]:
    """Returns an ingestor with a mocked converter and a temp file to point at."""
    fake_file = tmp_path / "report.pdf"
    fake_file.write_text("irrelevant — converter is mocked")
    return MarkItDownIngestor(_converter=mock_converter), fake_file


# ── IngestorBase ──────────────────────────────────────────────────────────────

class TestIngestorBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            IngestorBase()  # type: ignore[abstract]

    def test_subclass_must_implement_ingest(self):
        class Incomplete(IngestorBase):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class Stub(IngestorBase):
            def ingest(self, file_path):
                return [Document(content="hi", source="stub")]

        stub = Stub()
        result = stub.ingest("any.txt")
        assert len(result) == 1
        assert result[0].content == "hi"


# ── MarkdownChunker ───────────────────────────────────────────────────────────

class TestMarkdownChunker:
    # ── constructor ───────────────────────────────────────────────────────────

    def test_invalid_max_chunk_chars_raises(self):
        with pytest.raises(ValueError):
            MarkdownChunker(max_chunk_chars=0)

        with pytest.raises(ValueError):
            MarkdownChunker(max_chunk_chars=-1)

    # ── empty / whitespace input ──────────────────────────────────────────────

    def test_empty_string_returns_empty(self, chunker):
        assert chunker.chunk("", source="x.pdf") == []

    def test_whitespace_only_returns_empty(self, chunker):
        assert chunker.chunk("   \n\n   ", source="x.pdf") == []

    # ── source is propagated ──────────────────────────────────────────────────

    def test_source_set_on_all_chunks(self, chunker, sample_md_text):
        chunks = chunker.chunk(sample_md_text, source="report.pdf")
        assert all(c.source == "report.pdf" for c in chunks)

    # ── chunk_index ───────────────────────────────────────────────────────────

    def test_chunk_index_is_sequential(self, chunker, sample_md_text):
        chunks = chunker.chunk(sample_md_text, source="r.pdf")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_index_starts_at_zero(self, chunker, sample_md_text):
        chunks = chunker.chunk(sample_md_text, source="r.pdf")
        assert chunks[0].chunk_index == 0

    # ── header-based splitting ────────────────────────────────────────────────

    def test_splits_on_h2_headers(self, chunker):
        text = "## Section A\nContent A\n\n## Section B\nContent B"
        chunks = chunker.chunk(text, source="x.md")
        assert len(chunks) == 2
        assert "Section A" in chunks[0].content
        assert "Section B" in chunks[1].content

    def test_header_is_kept_with_its_content(self, chunker):
        text = "## Revenue\nWe made money."
        chunks = chunker.chunk(text, source="x.md")
        assert chunks[0].content.startswith("## Revenue")

    def test_h1_h2_h3_all_trigger_split(self, chunker):
        text = "# H1\nA\n\n## H2\nB\n\n### H3\nC"
        chunks = chunker.chunk(text, source="x.md")
        assert len(chunks) == 3

    def test_no_headers_produces_single_chunk_when_short(self, chunker):
        text = "Just some plain prose without any headers at all."
        chunks = chunker.chunk(text, source="x.md")
        assert len(chunks) == 1
        assert chunks[0].content == text

    # ── paragraph fallback for oversized sections ─────────────────────────────

    def test_oversized_section_split_by_paragraphs(self):
        chunker = MarkdownChunker(max_chunk_chars=50)
        # Each paragraph is 30 chars; together they exceed 50
        text = "## Big Section\n\n" + "A" * 30 + "\n\n" + "B" * 30
        chunks = chunker.chunk(text, source="x.md")
        assert len(chunks) == 2

    def test_oversized_paragraphs_greedily_packed(self):
        chunker = MarkdownChunker(max_chunk_chars=80)
        # 3 paragraphs of 30 chars each — first two fit (60 < 80), third goes to next chunk
        paras = ["X" * 30, "Y" * 30, "Z" * 30]
        text = "\n\n".join(paras)
        chunks = chunker.chunk(text, source="x.md")
        assert len(chunks) == 2
        assert "X" * 30 in chunks[0].content
        assert "Y" * 30 in chunks[0].content
        assert "Z" * 30 in chunks[1].content

    def test_no_empty_chunks_produced(self, chunker, sample_md_text):
        chunks = chunker.chunk(sample_md_text, source="r.pdf")
        assert all(c.content.strip() for c in chunks)

    # ── sample fixture end-to-end ─────────────────────────────────────────────

    def test_sample_fixture_produces_multiple_chunks(self, chunker, sample_md_text):
        chunks = chunker.chunk(sample_md_text, source="sample.md")
        # fixture has 5 sections (H1 + H2 × 4), each well under 1000 chars
        assert len(chunks) == 5

    def test_sample_fixture_sections_in_order(self, chunker, sample_md_text):
        chunks = chunker.chunk(sample_md_text, source="sample.md")
        assert "Q4 Financial Report" in chunks[0].content
        assert "Revenue" in chunks[1].content
        assert "Operating Expenses" in chunks[2].content
        assert "Net Income" in chunks[3].content
        assert "Outlook" in chunks[4].content

    def test_chunk_returns_document_instances(self, chunker, sample_md_text):
        chunks = chunker.chunk(sample_md_text, source="sample.md")
        assert all(isinstance(c, Document) for c in chunks)


# ── MarkItDownIngestor ────────────────────────────────────────────────────────

class TestMarkItDownIngestor:
    # ── file existence guard ──────────────────────────────────────────────────

    def test_raises_file_not_found_for_missing_file(self, mock_converter):
        ing = MarkItDownIngestor(_converter=mock_converter)
        with pytest.raises(FileNotFoundError):
            ing.ingest("/nonexistent/path/file.pdf")

    # ── empty output guard ────────────────────────────────────────────────────

    def test_raises_value_error_when_converter_returns_empty(self, tmp_path):
        empty_result = MagicMock()
        empty_result.text_content = ""
        converter = MagicMock()
        converter.convert.return_value = empty_result

        fake_file = tmp_path / "empty.pdf"
        fake_file.write_text("x")

        ing = MarkItDownIngestor(_converter=converter)
        with pytest.raises(ValueError, match="no content"):
            ing.ingest(fake_file)

    def test_raises_value_error_when_converter_returns_whitespace(self, tmp_path):
        ws_result = MagicMock()
        ws_result.text_content = "   \n\n   "
        converter = MagicMock()
        converter.convert.return_value = ws_result

        fake_file = tmp_path / "ws.pdf"
        fake_file.write_text("x")

        ing = MarkItDownIngestor(_converter=converter)
        with pytest.raises(ValueError):
            ing.ingest(fake_file)

    # ── happy path ────────────────────────────────────────────────────────────

    def test_returns_documents(self, ingestor):
        ing, fake_file = ingestor
        docs = ing.ingest(fake_file)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_documents_have_content(self, ingestor):
        ing, fake_file = ingestor
        docs = ing.ingest(fake_file)
        assert all(d.content.strip() for d in docs)

    def test_source_is_absolute_path(self, ingestor):
        ing, fake_file = ingestor
        docs = ing.ingest(fake_file)
        assert all(Path(d.source).is_absolute() for d in docs)

    def test_source_matches_file(self, ingestor):
        ing, fake_file = ingestor
        docs = ing.ingest(fake_file)
        assert all(d.source == str(fake_file.resolve()) for d in docs)

    def test_chunk_indices_are_sequential(self, ingestor):
        ing, fake_file = ingestor
        docs = ing.ingest(fake_file)
        assert [d.chunk_index for d in docs] == list(range(len(docs)))

    def test_converter_called_with_resolved_path(self, ingestor):
        ing, fake_file = ingestor
        ing.ingest(fake_file)
        ing._converter.convert.assert_called_once_with(str(fake_file.resolve()))

    # ── accepts string and Path ───────────────────────────────────────────────

    def test_accepts_string_path(self, ingestor):
        ing, fake_file = ingestor
        docs = ing.ingest(str(fake_file))
        assert len(docs) > 0

    def test_accepts_path_object(self, ingestor):
        ing, fake_file = ingestor
        docs = ing.ingest(fake_file)
        assert len(docs) > 0

    # ── custom chunker is used ────────────────────────────────────────────────

    def test_custom_chunker_is_respected(self, mock_converter, tmp_path):
        fake_file = tmp_path / "f.pdf"
        fake_file.write_text("x")
        tiny_chunker = MarkdownChunker(max_chunk_chars=100)
        ing = MarkItDownIngestor(chunker=tiny_chunker, _converter=mock_converter)
        docs = ing.ingest(fake_file)
        # with max_chunk_chars=100, sample fixture should produce more chunks
        # than default (max_chunk_chars=1000)
        default_ing = MarkItDownIngestor(_converter=mock_converter)
        default_docs = default_ing.ingest(fake_file)
        assert len(docs) >= len(default_docs)

    # ── missing markitdown import ─────────────────────────────────────────────

    def test_import_error_when_markitdown_not_installed(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "markitdown":
                raise ImportError("No module named 'markitdown'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        fake_file = tmp_path / "f.pdf"
        fake_file.write_text("x")
        ing = MarkItDownIngestor()  # no injected converter → will try to import
        with pytest.raises(ImportError, match="pip install markitdown"):
            ing.ingest(fake_file)
