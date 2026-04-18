"""
HybridRetriever — the main Phase 3 entry point.

Orchestrates three stages:
  1. Candidate retrieval — vector search + BM25, each returning candidate_k results
  2. RRF fusion         — merges both ranked lists into one without score normalisation
  3. Reranking          — cross-encoder scores the fused candidates (optional)

The reranker is optional. Without it, the RRF-fused ranking is returned directly.
With it, the cross-encoder re-scores the top candidates for much higher precision.

DataPacket contract:
  Input:  packet.query set; packet.expanded_queries may be non-empty (ignored for now,
          Phase 2 integration will use them for multi-query retrieval)
  Output: packet.documents = top_k Documents sorted by score descending
          packet.trace     = one TraceEntry appended, phase="retrieval"
"""

import time

from atomic_rag.models.base import EmbedderBase
from atomic_rag.retrieval.base import RetrieverBase
from atomic_rag.retrieval.bm25 import BM25Retriever
from atomic_rag.retrieval.fusion import reciprocal_rank_fusion
from atomic_rag.retrieval.reranker import CrossEncoderReranker
from atomic_rag.retrieval.vector_store import ChromaVectorStore
from atomic_rag.schema import DataPacket, Document, TraceEntry


class HybridRetriever(RetrieverBase):
    """
    Two-stage retriever: hybrid search (vector + BM25) then cross-encoder reranking.

    Args:
        embedder:     Converts query/document text to vectors. Required.
        vector_store: ChromaDB-backed store. Defaults to in-memory EphemeralClient.
        reranker:     Cross-encoder reranker. If None, RRF ranking is used directly.
        candidate_k:  Candidates fetched from each source before fusion/reranking.
                      Should be much larger than top_k (e.g. 50 for top_k=5).
    """

    def __init__(
        self,
        embedder: EmbedderBase,
        vector_store: ChromaVectorStore | None = None,
        reranker: CrossEncoderReranker | None = None,
        candidate_k: int = 50,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store or ChromaVectorStore()
        self.reranker = reranker
        self.candidate_k = candidate_k
        self._bm25 = BM25Retriever()

    def add_documents(self, documents: list[Document]) -> None:
        """
        Index documents in both the vector store and BM25.

        Embeddings are computed in batch for efficiency.
        Call this once before any retrieve() calls.
        """
        if not documents:
            return
        embeddings = self.embedder.embed_batch([d.content for d in documents])
        self.vector_store.add(documents, embeddings)
        self._bm25.add(documents)

    def retrieve(self, packet: DataPacket, top_k: int = 5) -> DataPacket:
        """
        Run hybrid retrieval and return a new DataPacket with documents populated.

        Stages:
          1. Embed query, search vector store for candidate_k results
          2. Search BM25 for candidate_k results
          3. Fuse with RRF → deduplicated, re-ranked candidates
          4. If reranker is set: cross-encode fused candidates → final top_k
             Otherwise: slice fused list to top_k directly

        TraceEntry details include counts at each stage for observability.
        """
        t0 = time.monotonic()

        query = packet.query

        # Stage 1: vector search
        query_embedding = self.embedder.embed(query)
        vector_hits = self.vector_store.search(query_embedding, self.candidate_k)

        # Stage 2: BM25 search
        bm25_hits = self._bm25.search(query, self.candidate_k)

        # Stage 3: RRF fusion
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
        candidates = fused[: self.candidate_k]

        # Stage 4: reranking (optional)
        if self.reranker and candidates:
            final_docs = self.reranker.rerank(query, candidates, top_k)
        else:
            final_docs = candidates[:top_k]

        duration_ms = (time.monotonic() - t0) * 1000
        trace_entry = TraceEntry(
            phase="retrieval",
            duration_ms=round(duration_ms, 2),
            details={
                "vector_hits": len(vector_hits),
                "bm25_hits": len(bm25_hits),
                "fused_candidates": len(candidates),
                "top_k": top_k,
                "reranked": self.reranker is not None,
            },
        )

        return packet.model_copy(update={"documents": final_docs}).with_trace(trace_entry)
