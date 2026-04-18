"""
AST-based Python code ingestor.

Chunks source files at semantic boundaries — functions, methods, and classes —
rather than at arbitrary character counts. This means every retrieved chunk is a
syntactically complete, self-contained unit that the LLM can read without context
from surrounding lines.

Chunking strategy per file:
  1. Module header  — docstring + imports (one chunk, gives context about the file)
  2. Top-level functions — one chunk each
  3. Classes — two levels:
       a. Class header chunk: class line + docstring (what is this class?)
       b. One chunk per method (what does this method do?)

Why two levels for classes?
  A query like "what does the DataPacket class do?" is best matched by the class
  header chunk. A query like "how does with_trace work?" is best matched by the
  method chunk. Keeping both means both query types hit the right chunk.

For multi-language support, tree-sitter can replace the ast module as the parser.
"""

import ast
from pathlib import Path
from typing import Iterator

from atomic_rag.ingestion.base import IngestorBase
from atomic_rag.schema import Document

_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", ".pytest_cache",
})


class CodeIngestor(IngestorBase):
    """
    Ingest Python source files using AST-based chunking.

    Implements IngestorBase.ingest() for single files, and adds
    ingest_directory() for walking an entire codebase.
    """

    def ingest(self, file_path: str | Path) -> list[Document]:
        """
        Parse a single .py file and return its chunks.

        Args:
            file_path: Path to a .py file.

        Returns:
            Ordered list of Documents (module header, functions, class headers, methods).

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the file is not a .py file or has a syntax error.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix != ".py":
            raise ValueError(
                f"CodeIngestor only supports .py files, got: {path.suffix!r}"
            )

        source = path.read_text(encoding="utf-8")
        return self._chunk_python(source, str(path))

    def ingest_directory(
        self,
        dir_path: str | Path,
        extensions: tuple[str, ...] = (".py",),
        ignore_dirs: frozenset[str] = _IGNORE_DIRS,
    ) -> list[Document]:
        """
        Walk a directory tree and ingest all matching source files.

        chunk_index is globally unique across all files so Documents
        can be stored in a flat list or vector store without collisions.

        Args:
            dir_path:    Root directory to walk.
            extensions:  File extensions to include. Default: .py only.
            ignore_dirs: Directory names to skip entirely.

        Returns:
            All chunks from all files, ordered by file path then chunk position.

        Raises:
            NotADirectoryError: if dir_path is not a directory.
        """
        root = Path(dir_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        all_docs: list[Document] = []
        global_idx = 0

        for file_path in self._walk(root, extensions, ignore_dirs):
            try:
                docs = self.ingest(file_path)
            except (ValueError, SyntaxError):
                # Skip files with syntax errors rather than aborting the whole run
                continue
            for doc in docs:
                all_docs.append(doc.model_copy(update={"chunk_index": global_idx}))
                global_idx += 1

        return all_docs

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _walk(
        self,
        root: Path,
        extensions: tuple[str, ...],
        ignore_dirs: frozenset[str],
    ) -> Iterator[Path]:
        for item in sorted(root.iterdir()):
            if item.is_dir() and item.name not in ignore_dirs:
                yield from self._walk(item, extensions, ignore_dirs)
            elif item.is_file() and item.suffix in extensions:
                yield item

    def _chunk_python(self, source: str, file_path: str) -> list[Document]:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Cannot parse {file_path}: {e}") from e

        lines = source.splitlines()
        chunks: list[Document] = []
        idx = 0

        # 1. Module header (docstring + imports)
        header = self._module_header(tree, lines)
        if header:
            chunks.append(Document(
                content=header,
                source=file_path,
                chunk_index=idx,
                metadata={"type": "module", "language": "python"},
            ))
            idx += 1

        # 2. Top-level functions and classes
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(Document(
                    content=self._node_source(node, lines),
                    source=file_path,
                    chunk_index=idx,
                    metadata={
                        "type": "function",
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "language": "python",
                    },
                ))
                idx += 1

            elif isinstance(node, ast.ClassDef):
                # 2a. Class header chunk (signature + docstring only)
                chunks.append(Document(
                    content=self._class_header(node, lines),
                    source=file_path,
                    chunk_index=idx,
                    metadata={
                        "type": "class",
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "language": "python",
                    },
                ))
                idx += 1

                # 2b. One chunk per method
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(Document(
                            content=self._node_source(item, lines),
                            source=file_path,
                            chunk_index=idx,
                            metadata={
                                "type": "method",
                                "name": item.name,
                                "class": node.name,
                                "start_line": item.lineno,
                                "end_line": item.end_lineno,
                                "language": "python",
                            },
                        ))
                        idx += 1

        return chunks

    def _node_source(self, node: ast.AST, lines: list[str]) -> str:
        return "\n".join(lines[node.lineno - 1 : node.end_lineno])

    def _module_header(self, tree: ast.Module, lines: list[str]) -> str:
        """Collect module docstring and all import statements at the top of the file."""
        parts: list[str] = []
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                parts.append(self._node_source(node, lines))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                parts.append(self._node_source(node, lines))
            else:
                break
        return "\n".join(parts).strip()

    def _class_header(self, node: ast.ClassDef, lines: list[str]) -> str:
        """
        Return the class definition line plus its docstring (if any).
        Excludes method bodies so methods are not duplicated in this chunk.
        """
        parts = [lines[node.lineno - 1]]  # class Foo(Bar):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            docstring_node = node.body[0]
            parts.extend(lines[docstring_node.lineno - 1 : docstring_node.end_lineno])
        return "\n".join(parts).strip()
