"""
Abstract interface for all retrieval implementations.

A retriever takes a DataPacket (with query set) and returns a new DataPacket
with documents populated and scored. The caller decides top_k.

Retrievers must also implement add_documents() for indexing. This is called
once at index time; retrieve() is called at query time per request.
"""

from abc import ABC, abstractmethod

from atomic_rag.schema import DataPacket, Document


class RetrieverBase(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """
        Index documents for retrieval.

        Called once before any retrieve() calls. Implementations should
        embed and store documents in whatever backing stores they use.

        Args:
            documents: Documents to index. Must have content and id set.
        """

    @abstractmethod
    def retrieve(self, packet: DataPacket, top_k: int = 5) -> DataPacket:
        """
        Search for documents relevant to packet.query.

        Returns a new DataPacket (never mutates input) with:
          - documents: top_k Documents sorted by score descending
          - trace: one TraceEntry appended with phase="retrieval"

        Args:
            packet: Input packet. Uses packet.query for search.
                    packet.expanded_queries will be used when Phase 2 integrates.
            top_k:  Number of documents to return after reranking.
        """
