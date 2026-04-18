"""
Tests for atomic_rag.ingestion.CodeIngestor — AST-based Python chunking.

No mocking needed: the ast module is stdlib and always available.
Tests use an inline sample source string rather than real project files
so they don't break if file contents change.
"""

from pathlib import Path

import pytest

from atomic_rag.ingestion import CodeIngestor
from atomic_rag.schema import Document

# ── Sample source used across all chunker tests ───────────────────────────────

SAMPLE_PYTHON = '''\
"""Module docstring for sample."""
import os
from pathlib import Path


def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


def farewell(name: str) -> str:
    return f"Goodbye, {name}!"


class Counter:
    """A simple counter class."""

    def __init__(self, start: int = 0) -> None:
        self.value = start

    def increment(self) -> None:
        self.value += 1

    def reset(self) -> None:
        self.value = 0
'''

# Expected chunk layout for SAMPLE_PYTHON:
#  0 — module header  (docstring + imports)
#  1 — greet          (function)
#  2 — farewell       (function)
#  3 — Counter        (class header)
#  4 — __init__       (method)
#  5 — increment      (method)
#  6 — reset          (method)
EXPECTED_CHUNK_COUNT = 7


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ingestor() -> CodeIngestor:
    return CodeIngestor()


@pytest.fixture
def sample_py_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE_PYTHON)
    return f


@pytest.fixture
def sample_chunks(ingestor: CodeIngestor, sample_py_file: Path) -> list[Document]:
    return ingestor.ingest(sample_py_file)


# ── Error cases ───────────────────────────────────────────────────────────────

class TestCodeIngestorErrors:
    def test_raises_file_not_found(self, ingestor):
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("/nonexistent/file.py")

    def test_raises_for_non_python_file(self, ingestor, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Hello")
        with pytest.raises(ValueError, match=".py"):
            ingestor.ingest(f)

    def test_raises_for_syntax_error(self, ingestor, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def foo(:\n    pass")
        with pytest.raises(ValueError, match="Cannot parse"):
            ingestor.ingest(f)

    def test_raises_not_a_directory(self, ingestor, sample_py_file):
        with pytest.raises(NotADirectoryError):
            ingestor.ingest_directory(sample_py_file)


# ── Chunk count and ordering ──────────────────────────────────────────────────

class TestChunkLayout:
    def test_produces_correct_number_of_chunks(self, sample_chunks):
        assert len(sample_chunks) == EXPECTED_CHUNK_COUNT

    def test_chunk_indices_are_sequential_from_zero(self, sample_chunks):
        assert [c.chunk_index for c in sample_chunks] == list(range(EXPECTED_CHUNK_COUNT))

    def test_all_chunks_are_document_instances(self, sample_chunks):
        assert all(isinstance(c, Document) for c in sample_chunks)

    def test_no_empty_chunks(self, sample_chunks):
        assert all(c.content.strip() for c in sample_chunks)


# ── Module header chunk ───────────────────────────────────────────────────────

class TestModuleHeaderChunk:
    def test_first_chunk_is_module_type(self, sample_chunks):
        assert sample_chunks[0].metadata["type"] == "module"

    def test_module_chunk_contains_docstring(self, sample_chunks):
        assert "Module docstring for sample" in sample_chunks[0].content

    def test_module_chunk_contains_imports(self, sample_chunks):
        content = sample_chunks[0].content
        assert "import os" in content
        assert "from pathlib import Path" in content

    def test_no_module_chunk_for_file_with_no_header(self, ingestor, tmp_path):
        f = tmp_path / "bare.py"
        f.write_text("x = 1\n\ndef foo():\n    pass\n")
        chunks = ingestor.ingest(f)
        # x = 1 stops the header scan, so no module chunk
        types = [c.metadata["type"] for c in chunks]
        assert "module" not in types


# ── Function chunks ───────────────────────────────────────────────────────────

class TestFunctionChunks:
    def test_greet_chunk_metadata(self, sample_chunks):
        greet = sample_chunks[1]
        assert greet.metadata["type"] == "function"
        assert greet.metadata["name"] == "greet"
        assert greet.metadata["language"] == "python"

    def test_greet_chunk_content(self, sample_chunks):
        assert "def greet" in sample_chunks[1].content
        assert "Hello" in sample_chunks[1].content

    def test_farewell_chunk(self, sample_chunks):
        assert sample_chunks[2].metadata["name"] == "farewell"

    def test_function_chunk_has_line_numbers(self, sample_chunks):
        greet = sample_chunks[1]
        assert "start_line" in greet.metadata
        assert "end_line" in greet.metadata
        assert greet.metadata["start_line"] < greet.metadata["end_line"]


# ── Class header chunk ────────────────────────────────────────────────────────

class TestClassHeaderChunk:
    def test_class_chunk_type(self, sample_chunks):
        counter = sample_chunks[3]
        assert counter.metadata["type"] == "class"
        assert counter.metadata["name"] == "Counter"

    def test_class_header_contains_definition_line(self, sample_chunks):
        assert "class Counter" in sample_chunks[3].content

    def test_class_header_contains_docstring(self, sample_chunks):
        assert "A simple counter class" in sample_chunks[3].content

    def test_class_header_does_not_contain_method_bodies(self, sample_chunks):
        # The class header chunk should not include method implementations
        content = sample_chunks[3].content
        assert "self.value += 1" not in content
        assert "self.value = 0" not in content

    def test_class_without_docstring(self, ingestor, tmp_path):
        f = tmp_path / "nodoc.py"
        f.write_text("class Foo:\n    def bar(self): pass\n")
        chunks = ingestor.ingest(f)
        class_chunk = next(c for c in chunks if c.metadata["type"] == "class")
        assert "class Foo" in class_chunk.content


# ── Method chunks ─────────────────────────────────────────────────────────────

class TestMethodChunks:
    def test_init_method_chunk(self, sample_chunks):
        init = sample_chunks[4]
        assert init.metadata["type"] == "method"
        assert init.metadata["name"] == "__init__"
        assert init.metadata["class"] == "Counter"

    def test_increment_method_chunk(self, sample_chunks):
        inc = sample_chunks[5]
        assert inc.metadata["name"] == "increment"
        assert "self.value += 1" in inc.content

    def test_reset_method_chunk(self, sample_chunks):
        reset = sample_chunks[6]
        assert reset.metadata["name"] == "reset"
        assert "self.value = 0" in reset.content

    def test_method_chunk_has_class_in_metadata(self, sample_chunks):
        methods = [c for c in sample_chunks if c.metadata["type"] == "method"]
        assert all(c.metadata["class"] == "Counter" for c in methods)


# ── Source field ──────────────────────────────────────────────────────────────

class TestSourceField:
    def test_source_is_absolute_path(self, sample_chunks, sample_py_file):
        assert all(Path(c.source).is_absolute() for c in sample_chunks)

    def test_source_matches_input_file(self, sample_chunks, sample_py_file):
        assert all(c.source == str(sample_py_file.resolve()) for c in sample_chunks)


# ── ingest_directory ──────────────────────────────────────────────────────────

class TestIngestDirectory:
    def test_finds_python_files(self, ingestor, tmp_path):
        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("def bar(): pass\n")
        (tmp_path / "README.md").write_text("# Ignore me")
        docs = ingestor.ingest_directory(tmp_path)
        sources = {Path(d.source).name for d in docs}
        assert "a.py" in sources
        assert "b.py" in sources
        assert "README.md" not in sources

    def test_global_chunk_indices_are_sequential(self, ingestor, tmp_path):
        (tmp_path / "a.py").write_text("def foo(): pass\ndef bar(): pass\n")
        (tmp_path / "b.py").write_text("def baz(): pass\n")
        docs = ingestor.ingest_directory(tmp_path)
        assert [d.chunk_index for d in docs] == list(range(len(docs)))

    def test_ignores_pycache(self, ingestor, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "hidden.py").write_text("def secret(): pass\n")
        (tmp_path / "visible.py").write_text("def visible(): pass\n")
        docs = ingestor.ingest_directory(tmp_path)
        sources = {Path(d.source).name for d in docs}
        assert "hidden.py" not in sources
        assert "visible.py" in sources

    def test_ignores_venv(self, ingestor, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "site.py").write_text("x = 1\n")
        (tmp_path / "app.py").write_text("def run(): pass\n")
        docs = ingestor.ingest_directory(tmp_path)
        sources = {Path(d.source).name for d in docs}
        assert "site.py" not in sources

    def test_walks_subdirectories(self, ingestor, tmp_path):
        sub = tmp_path / "subpackage"
        sub.mkdir()
        (tmp_path / "top.py").write_text("def top(): pass\n")
        (sub / "nested.py").write_text("def nested(): pass\n")
        docs = ingestor.ingest_directory(tmp_path)
        sources = {Path(d.source).name for d in docs}
        assert "top.py" in sources
        assert "nested.py" in sources

    def test_skips_files_with_syntax_errors(self, ingestor, tmp_path):
        (tmp_path / "good.py").write_text("def ok(): pass\n")
        (tmp_path / "bad.py").write_text("def broken(:\n    pass")
        # Should not raise — bad.py is skipped
        docs = ingestor.ingest_directory(tmp_path)
        sources = {Path(d.source).name for d in docs}
        assert "good.py" in sources
        assert "bad.py" not in sources

    def test_empty_directory_returns_empty_list(self, ingestor, tmp_path):
        assert ingestor.ingest_directory(tmp_path) == []
