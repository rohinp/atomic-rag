"""
Step 2 — Ask questions about the atomic-rag codebase.

Ingests the codebase on startup (in-memory index, rebuilt each run), then
answers questions using query expansion + hybrid retrieval + context compression.

Requires Ollama running locally with these models pulled:
    ollama pull nomic-embed-text
    ollama pull llama3.2:3b

Run:
    python examples/code_qa/query.py "how does DataPacket work?"
    python examples/code_qa/query.py --hyde "what does the MarkdownChunker do?"
    python examples/code_qa/query.py --multi-query "how does retrieval work?"
    python examples/code_qa/query.py  # interactive mode

To use OpenAI instead, edit config.py.

Status — grows as phases ship:
    [done]    Phase 1: ingestion  (CodeIngestor)
    [done]    Phase 2: query expansion (HyDEExpander, MultiQueryExpander)
    [done]    Phase 3: retrieval  (HybridRetriever, vector + BM25)
    [done]    Phase 4: context compression (SentenceCompressor)
    [planned] Phase 5: LLM answer generation with C-RAG
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def build_index():
    from atomic_rag.context import SentenceCompressor
    from atomic_rag.ingestion import CodeIngestor
    from atomic_rag.retrieval import HybridRetriever

    from examples.code_qa.config import EMBEDDER

    print("Building index... ", end="", flush=True)
    t0 = time.monotonic()

    docs = CodeIngestor().ingest_directory(ROOT / "atomic_rag")
    retriever = HybridRetriever(embedder=EMBEDDER)
    retriever.add_documents(docs)
    compressor = SentenceCompressor(embedder=EMBEDDER, threshold=0.45)

    elapsed = time.monotonic() - t0
    print(f"{len(docs)} chunks indexed in {elapsed:.1f}s")
    return retriever, compressor


def run_query(
    retriever,
    compressor,
    query: str,
    top_k: int = 5,
    expansion: str = "none",
) -> None:
    from atomic_rag.schema import DataPacket

    packet = DataPacket(query=query)

    # Phase 2: query expansion (if requested)
    if expansion != "none":
        from examples.code_qa.config import CHAT_MODEL

        if expansion == "hyde":
            from atomic_rag.query import HyDEExpander
            expander = HyDEExpander(chat_model=CHAT_MODEL)
        else:  # multi-query
            from atomic_rag.query import MultiQueryExpander
            expander = MultiQueryExpander(chat_model=CHAT_MODEL, n_queries=3)

        packet = expander.expand(packet)
        q_trace = next(t for t in packet.trace if t.phase == "query_expansion")
        strategy = q_trace.details["strategy"]
        n_expanded = q_trace.details["expanded_count"]
        print(f"Expansion: {strategy}  ({n_expanded} queries, {q_trace.duration_ms:.0f}ms)")

    # Phase 3: retrieval
    packet = retriever.retrieve(packet, top_k=top_k)

    # Phase 4: context compression
    packet = compressor.compress(packet)

    r_trace = next(t for t in packet.trace if t.phase == "retrieval")
    c_trace = next(t for t in packet.trace if t.phase == "context")

    print(f"\nQuery    : {query}")
    print(f"Retrieval: {r_trace.duration_ms:.0f}ms  "
          f"(queries={r_trace.details['queries_used']}, "
          f"vector={r_trace.details['vector_hits']}, "
          f"bm25={r_trace.details['bm25_hits']})")
    print(f"Compress : {c_trace.duration_ms:.0f}ms  "
          f"({c_trace.details['sentences_before']} → "
          f"{c_trace.details['sentences_after']} sentences, "
          f"{c_trace.details['reduction_pct']}% removed)")
    print(f"\n--- Context for LLM ---\n")
    print(packet.context or "(empty — no documents above threshold)")
    print(f"\n--- End context ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the atomic-rag codebase.")
    parser.add_argument("query", nargs="*", help="Question to ask")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hyde", action="store_true", help="Use HyDE query expansion")
    group.add_argument("--multi-query", action="store_true", help="Use multi-query expansion")
    args = parser.parse_args()

    expansion = "none"
    if args.hyde:
        expansion = "hyde"
    elif args.multi_query:
        expansion = "multi_query"

    try:
        retriever, compressor = build_index()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running and models are pulled:")
        print("  ollama pull nomic-embed-text")
        print("  ollama pull llama3.2:3b")
        sys.exit(1)

    if args.query:
        run_query(retriever, compressor, " ".join(args.query), expansion=expansion)
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
                run_query(retriever, compressor, q, expansion=expansion)


if __name__ == "__main__":
    main()
