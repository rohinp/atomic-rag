"""
Abstract base for pipeline evaluation strategies.

Each evaluator takes a DataPacket with answer + context set and returns a
new packet with eval_scores partially or fully populated.  Multiple evaluators
can be chained — each one fills in the scores it owns.

DataPacket contract:
  Input:  packet.query, packet.context, packet.answer (Phase 5 output)
  Output: packet.eval_scores updated; one TraceEntry appended (phase="evaluation")
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from atomic_rag.schema import DataPacket


class PipelineEvalBase(ABC):
    """Contract for all RAG pipeline evaluators."""

    @abstractmethod
    def score(self, packet: DataPacket) -> DataPacket:
        """
        Compute evaluation metrics and return a new packet with eval_scores set.

        Rules:
        - Never mutate the input packet.
        - Only overwrite the specific eval_scores fields this evaluator owns;
          leave others as they are so evaluators can be chained.
        - Always append a TraceEntry (phase="evaluation") even on partial success.
        """
