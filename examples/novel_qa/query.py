"""
Step 2 — Ask questions about The War of the Worlds by H. G. Wells.

Full pipeline: query expansion → hybrid retrieval → context compression →
C-RAG evaluation → grounded LLM answer.

The novel is downloaded automatically on first run from Project Gutenberg
(public domain). Subsequent runs use the cached local copy.

Requires Ollama running locally with these models pulled:
    ollama pull nomic-embed-text
    ollama pull llama3.2:3b

Run:
    python examples/novel_qa/query.py "what are the Martians like?"
    python examples/novel_qa/query.py --hyde "how do humans fight back?"
    python examples/novel_qa/query.py --multi-query "what happens in the end?"
    python examples/novel_qa/query.py --no-answer   # show context only
    python examples/novel_qa/query.py --verbose     # show docs, context, eval
    python examples/novel_qa/query.py               # interactive mode

To use OpenAI instead, edit config.py.

Status:
    [done] Phase 1: ingestion  (MarkItDownIngestor)
    [done] Phase 2: query expansion (HyDEExpander, MultiQueryExpander)
    [done] Phase 3: retrieval  (HybridRetriever, vector + BM25)
    [done] Phase 4: context compression (SentenceCompressor)
    [done] Phase 5: LLM answer generation with C-RAG (AgentRunner)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.novel_qa.ingest import NOVEL_PATH, download_novel  # noqa: E402


def build_index():
    from atomic_rag.agent import AgentRunner, LLMEvaluator, LLMGenerator
    from atomic_rag.context import SentenceCompressor
    from atomic_rag.ingestion import MarkItDownIngestor
    from atomic_rag.ingestion.chunker import MarkdownChunker
    from atomic_rag.retrieval import HybridRetriever

    from examples.novel_qa.config import CHAT_MODEL, EMBEDDER

    # Download novel on first run
    path = download_novel(dest=NOVEL_PATH)

    print("Building index... ", end="", flush=True)
    t0 = time.monotonic()

    ingestor = MarkItDownIngestor(chunker=MarkdownChunker(max_chunk_chars=800))
    docs = ingestor.ingest(path)

    retriever = HybridRetriever(embedder=EMBEDDER)
    retriever.add_documents(docs)

    compressor = SentenceCompressor(embedder=EMBEDDER, threshold=0.45)

    runner = AgentRunner(
        evaluator=LLMEvaluator(chat_model=CHAT_MODEL),
        generator=LLMGenerator(chat_model=CHAT_MODEL),
        # 0.2 is intentionally permissive for small local models (llama3.2:3b).
        # Raise to 0.5+ when using a capable API model.
        threshold=0.2,
    )

    elapsed = time.monotonic() - t0
    print(f"{len(docs)} chunks indexed in {elapsed:.1f}s")
    return retriever, compressor, runner


def run_query(
    retriever,
    compressor,
    runner,
    query: str,
    top_k: int = 5,
    expansion: str = "none",
    show_answer: bool = True,
    verbose: bool = False,
) -> None:
    from atomic_rag.schema import DataPacket

    packet = DataPacket(query=query)

    # Phase 2: query expansion (if requested)
    if expansion != "none":
        from examples.novel_qa.config import CHAT_MODEL

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

        if verbose and packet.expanded_queries:
            print("  Expanded queries:")
            for i, eq in enumerate(packet.expanded_queries):
                print(f"    [{i}] {eq[:120]}")

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

    if verbose:
        print("\n--- Retrieved chunks ---")
        for doc in packet.documents:
            print(f"  [{doc.score:.3f}] chunk={doc.chunk_index}")
            print(f"          {doc.content[:120].replace(chr(10), ' ')}...")
        print()
        print("--- Compressed context ---")
        print(packet.context or "(empty)")
        print("--- End context ---\n")

    if not show_answer:
        if not verbose:
            print(f"\n--- Context for LLM ---\n")
            print(packet.context or "(empty — no documents above threshold)")
            print(f"\n--- End context ---")
        return

    # Phase 5: C-RAG answer generation
    packet = runner.run(packet)
    a_trace = next(t for t in packet.trace if t.phase == "agent")

    print(f"Agent    : {a_trace.duration_ms:.0f}ms  "
          f"(eval_score={a_trace.details['eval_score']:.2f}, "
          f"threshold={a_trace.details['threshold']}, "
          f"fallback={a_trace.details['fallback']})")

    if verbose and "eval_raw" in a_trace.details:
        print(f"  Evaluator raw response: {a_trace.details['eval_raw']!r}")

    print(f"\n--- Answer ---\n")
    print(packet.answer)
    print(f"\n--- End answer ---")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about The War of the Worlds."
    )
    parser.add_argument("query", nargs="*", help="Question to ask")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hyde", action="store_true", help="Use HyDE query expansion")
    group.add_argument("--multi-query", action="store_true", help="Use multi-query expansion")
    parser.add_argument("--no-answer", action="store_true", help="Show context only, skip LLM answer")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show retrieved chunks, compressed context, and raw evaluator response")
    args = parser.parse_args()

    expansion = "none"
    if args.hyde:
        expansion = "hyde"
    elif args.multi_query:
        expansion = "multi_query"

    show_answer = not args.no_answer
    verbose = args.verbose

    try:
        retriever, compressor, runner = build_index()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running and models are pulled:")
        print("  ollama pull nomic-embed-text")
        print("  ollama pull llama3.2:3b")
        sys.exit(1)

    if args.query:
        run_query(retriever, compressor, runner, " ".join(args.query),
                  top_k=8, expansion=expansion, show_answer=show_answer, verbose=verbose)
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
                run_query(retriever, compressor, runner, q,
                          top_k=8, expansion=expansion, show_answer=show_answer, verbose=verbose)


if __name__ == "__main__":
    main()
