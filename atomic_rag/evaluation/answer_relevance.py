"""
EmbeddingAnswerRelevance — measures whether the answer addresses the question.

Algorithm:
  1. Embed the original query.
  2. Embed the answer.
  3. Score = cosine_similarity(query_embedding, answer_embedding).

A high score means the answer is topically close to the question.  A low score
means the answer drifted off-topic — e.g. the context was about the right
subject but the generator answered a tangential sub-question.

This is a cheaper, embedding-only approximation of the Ragas answer_relevancy
metric (which generates N synthetic questions from the answer and measures how
closely they match the original).  The cosine approach is faster (2 embed
calls vs. N LLM calls + N embed calls) and sufficient for most use cases.

Reuses the same embedder as retrieval and context compression to keep the
vector space consistent — "close to the query" means the same thing everywhere
in the pipeline.
"""

from __future__ import annotations

import math
import time

from atomic_rag.models.base import EmbedderBase
from atomic_rag.schema import DataPacket, TraceEntry

from .base import PipelineEvalBase


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingAnswerRelevance(PipelineEvalBase):
    """
    Scores answer relevance as cosine similarity between query and answer
    embeddings.

    Parameters
    ----------
    embedder:
        Any EmbedderBase implementation.  Use the same embedder as the
        retrieval stage for a consistent vector space.
    """

    def __init__(self, embedder: EmbedderBase) -> None:
        self._embedder = embedder

    def score(self, packet: DataPacket) -> DataPacket:
        t0 = time.monotonic()

        query = packet.query.strip()
        answer = packet.answer.strip()

        if not query or not answer:
            relevance = 0.0
        else:
            q_emb, a_emb = self._embedder.embed_batch([query, answer])
            relevance = max(0.0, _cosine(q_emb, a_emb))

        duration_ms = (time.monotonic() - t0) * 1000

        new_scores = packet.eval_scores.model_copy(
            update={"answer_relevance": round(relevance, 4)}
        )
        entry = TraceEntry(
            phase="evaluation",
            duration_ms=round(duration_ms, 2),
            details={
                "metric": "answer_relevance",
                "score": round(relevance, 4),
            },
        )
        return packet.model_copy(update={"eval_scores": new_scores}).with_trace(entry)
