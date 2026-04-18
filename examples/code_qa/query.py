"""
Step 2 — Ask questions about the codebase.

This file grows as phases ship:

  Phase 3 (retrieval) ── vector store + BM25 search against ingested chunks
  Phase 4 (context)   ── compress retrieved chunks before sending to LLM
  Phase 2 (query)     ── expand the query with HyDE / multi-query
  Phase 5 (agent)     ── LLM generates the answer, C-RAG catches hallucinations

Current status: Phase 3 not yet built.
Run ingest.py to see what chunks would be searched.
"""


def main() -> None:
    print("Phase 3 (hybrid retrieval) not yet implemented.")
    print()
    print("To see what the index looks like, run:")
    print("  python examples/code_qa/ingest.py")


if __name__ == "__main__":
    main()
