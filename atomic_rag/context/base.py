"""
Abstract interface for context compression.

A compressor takes a DataPacket with documents populated (Phase 3 output)
and returns a new DataPacket with context populated — a compressed string
ready to be passed to the LLM.
"""

from abc import ABC, abstractmethod

from atomic_rag.schema import DataPacket


class CompressorBase(ABC):
    @abstractmethod
    def compress(self, packet: DataPacket) -> DataPacket:
        """
        Compress retrieved documents into a context string.

        Args:
            packet: Must have query and documents set (Phase 3 output).

        Returns:
            New DataPacket (never mutates input) with:
              - context: compressed string for the LLM
              - trace:   one TraceEntry appended, phase="context"
        """
