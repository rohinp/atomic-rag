"""
Abstract base for query expansion strategies.

Each strategy takes a DataPacket with `query` set and returns a new packet
with `expanded_queries` populated.  Downstream retrieval checks `expanded_queries`
first and falls back to the raw `query` if it is empty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from atomic_rag.schema import DataPacket


class QueryExpansionBase(ABC):
    """Contract for all query expansion strategies."""

    @abstractmethod
    def expand(self, packet: DataPacket) -> DataPacket:
        """
        Expand the query in *packet* and return a new packet with
        `expanded_queries` and one `TraceEntry` (phase="query_expansion") appended.

        Rules:
        - Never mutate the input packet.
        - Always append a TraceEntry even when expansion fails gracefully.
        - `expanded_queries` must contain at least one string on success.
        """
