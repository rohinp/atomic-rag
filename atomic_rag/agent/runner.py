"""
AgentRunner — the Phase 5 C-RAG orchestrator.

Corrective RAG (C-RAG) adds a quality gate before answer generation:

  1. Evaluate: score how well the retrieved context can answer the query.
  2. If score >= threshold: generate a grounded answer.
  3. If score < threshold: trigger a fallback (configurable refusal message)
     instead of generating a potentially hallucinated response.

This design ensures the pipeline degrades gracefully when retrieval fails —
the user receives an honest "I don't have enough information" rather than a
confident but wrong answer.

Reference:
  Yan et al., "Corrective Retrieval Augmented Generation" (2024).
  https://arxiv.org/abs/2401.15884

DataPacket contract:
  Input:  packet.query, packet.context (Phase 4 output)
  Output: packet.answer populated; one TraceEntry appended (phase="agent")
"""

from __future__ import annotations

import time

from atomic_rag.schema import DataPacket, TraceEntry

from .base import EvaluatorBase, GeneratorBase

_DEFAULT_FALLBACK = (
    "I don't have enough information in the retrieved context to answer this question. "
    "Try rephrasing your query or expanding the search."
)


class AgentRunner:
    """
    Orchestrates the C-RAG loop: evaluate → generate or fallback.

    Parameters
    ----------
    evaluator:
        Scores context quality.  Any EvaluatorBase implementation.
    generator:
        Produces the final answer.  Any GeneratorBase implementation.
    threshold:
        Minimum evaluator score to proceed with generation.
        Below this, the fallback message is used instead.
    fallback_message:
        Returned as `answer` when context quality is insufficient.
    """

    def __init__(
        self,
        evaluator: EvaluatorBase,
        generator: GeneratorBase,
        threshold: float = 0.5,
        fallback_message: str = _DEFAULT_FALLBACK,
    ) -> None:
        self.evaluator = evaluator
        self.generator = generator
        self.threshold = threshold
        self.fallback_message = fallback_message

    def run(self, packet: DataPacket) -> DataPacket:
        """
        Evaluate context quality and generate an answer or trigger fallback.

        Returns a new DataPacket with `answer` set and one TraceEntry appended.
        Input packet is never mutated.
        """
        t0 = time.monotonic()

        context = packet.context
        query = packet.query

        # Stage 1: evaluate context quality
        eval_score = self.evaluator.evaluate(query, context)

        # Stage 2: generate or fallback
        if eval_score >= self.threshold:
            answer = self.generator.generate(query, context)
            used_fallback = False
        else:
            answer = self.fallback_message
            used_fallback = True

        duration_ms = (time.monotonic() - t0) * 1000
        details: dict = {
            "eval_score": round(eval_score, 4),
            "threshold": self.threshold,
            "fallback": used_fallback,
            "answer_length": len(answer),
        }
        # Capture raw evaluator response when available — useful for debugging
        # why the score landed where it did (small models sometimes ignore format).
        if hasattr(self.evaluator, "last_raw_response"):
            details["eval_raw"] = self.evaluator.last_raw_response

        trace_entry = TraceEntry(
            phase="agent",
            duration_ms=round(duration_ms, 2),
            details=details,
        )

        return packet.model_copy(update={"answer": answer}).with_trace(trace_entry)
