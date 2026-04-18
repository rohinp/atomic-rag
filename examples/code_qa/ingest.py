"""
Step 1 — Ingest the atomic-rag codebase.

Dogfoods the library against its own source code. Produces the chunk
index that later phases will search against.

Run:
    python examples/code_qa/ingest.py [path/to/repo]

If no path is given, ingests the atomic-rag source directory itself.
"""

import sys
from collections import Counter
from pathlib import Path

from atomic_rag.ingestion import CodeIngestor

ROOT = Path(__file__).parent.parent.parent  # atomic-rag repo root


def main(target: Path) -> None:
    ingestor = CodeIngestor()
    print(f"Ingesting: {target}\n")

    docs = ingestor.ingest_directory(target)

    if not docs:
        print("No Python files found.")
        return

    # Summary
    files = {Path(d.source) for d in docs}
    type_counts = Counter(d.metadata["type"] for d in docs)

    print(f"Files:   {len(files)}")
    print(f"Chunks:  {len(docs)}")
    print(f"  module headers : {type_counts['module']}")
    print(f"  functions      : {type_counts['function']}")
    print(f"  classes        : {type_counts['class']}")
    print(f"  methods        : {type_counts['method']}")
    print()

    # Preview first 8 chunks
    print("Preview (first 8 chunks):")
    print("-" * 60)
    for doc in docs[:8]:
        meta = doc.metadata
        label = meta.get("name", meta["type"])
        loc = ""
        if "start_line" in meta:
            loc = f"  L{meta['start_line']}–{meta['end_line']}"
        file_name = Path(doc.source).relative_to(ROOT)
        print(f"  [{doc.chunk_index:>3}] {meta['type']:<8}  {label:<30} {file_name}{loc}")

    if len(docs) > 8:
        print(f"  ... and {len(docs) - 8} more chunks")

    print()
    print("Next: Phase 3 (hybrid retrieval) will make these chunks searchable.")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "atomic_rag"
    main(target)
