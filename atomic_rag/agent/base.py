"""
Abstract base classes for Phase 5 — Agentic Reasoning.

Two contracts:
- EvaluatorBase: scores how well the context can answer the query (0.0–1.0)
- GeneratorBase: produces a final answer given query + context

AgentRunner composes them into the C-RAG loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EvaluatorBase(ABC):
    """
    Scores the quality of the retrieved context for answering the query.

    Returns a float in [0.0, 1.0]:
      1.0 — context fully answers the query
      0.0 — context is irrelevant or contains no useful information

    Implementations may call an LLM, use a cross-encoder, or apply heuristics.
    The score is compared against AgentRunner.threshold to decide whether to
    generate an answer or trigger a fallback.
    """

    @abstractmethod
    def evaluate(self, query: str, context: str) -> float:
        """Return a relevance score in [0.0, 1.0]."""


class GeneratorBase(ABC):
    """
    Generates a final answer from a query and a context string.

    The context is the compressed output of Phase 4 — it is already filtered
    to the most relevant sentences.  The generator should ground its answer
    strictly in the context and not hallucinate beyond it.
    """

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Return the answer string."""
