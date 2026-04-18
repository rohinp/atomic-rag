"""
Step 2 — Ask questions about the atomic-rag codebase.

Ingests the codebase on startup (in-memory index, rebuilt each run), then
answers questions using hybrid retrieval (vector + BM25 + optional reranking).

Requires Ollama running locally with these models pulled:
    ollama pull nomic-embed-text
    ollama pull llama3.2:3b

Run:
    python examples/code_qa/query.py "how does DataPacket work?"
    python examples/code_qa/query.py "what does the MarkdownChunker do?"
    python examples/code_qa/query.py  # interactive mode

To use OpenAI instead, edit config.py.

Status — grows as phases ship:
    [done]    Phase 1: ingestion  (CodeIngestor)
    [done]    Phase 3: retrieval  (HybridRetriever, vector + BM25)
    [planned] Phase 4: context compression
    [planned] Phase 2: query expansion (HyDE / multi-query)
    [planned] Phase 5: LLM answer generation with C-RAG
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def build_index():
    from atomic_rag.ingestion import CodeIngestor
    from atomic_rag.retrieval import HybridRetriever

    from examples.code_qa.config import EMBEDDER

    print("Building index... ", end="", flush=True)
    t0 = time.monotonic()

    docs = CodeIngestor().ingest_directory(ROOT / "atomic_rag")
    retriever = HybridRetriever(embedder=EMBEDDER)
    retriever.add_documents(docs)

    elapsed = time.monotonic() - t0
    print(f"{len(docs)} chunks indexed in {elapsed:.1f}s")
    return retriever


def run_query(retriever, query: str, top_k: int = 5) -> None:
    from atomic_rag.schema import DataPacket

    packet = DataPacket(query=query)
    result = retriever.retrieve(packet, top_k=top_k)

    trace = result.trace[-1]
    print(f"\nQuery : {query}")
    print(f"Time  : {trace.duration_ms:.0f}ms  "
          f"(vector={trace.details['vector_hits']}, "
          f"bm25={trace.details['bm25_hits']}, "
          f"fused={trace.details['fused_candidates']})")
    print(f"\nTop {len(result.documents)} results:")
    print("-" * 60)
    for i, doc in enumerate(result.documents, 1):
        meta = doc.metadata
        label = meta.get("name", meta.get("type", "?"))
        loc = f"L{meta['start_line']}–{meta['end_line']}" if "start_line" in meta else ""
        file_name = Path(doc.source).relative_to(ROOT)
        print(f"\n[{i}] {meta.get('type', '?'):8}  {label}  ({file_name} {loc})  score={doc.score:.4f}")
        print(f"    {doc.content[:200].replace(chr(10), ' ')}")


def main() -> None:
    try:
        retriever = build_index()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running and models are pulled:")
        print("  ollama pull nomic-embed-text")
        print("  ollama pull llama3.2:3b")
        sys.exit(1)

    if len(sys.argv) > 1:
        run_query(retriever, " ".join(sys.argv[1:]))
    else:
        print('\nInteractive mode. Type "quit" to exit.\n')
        while True:
            try:
                q = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                run_query(retriever, q)


if __name__ == "__main__":
    main()
