"""
Inter-module data contract for atomic-rag.

Every phase in the pipeline consumes a DataPacket and returns a new DataPacket
with its output fields populated. No phase mutates the input — always return a
copy (use packet.model_copy(update={...})).

Flow:
    query (str)
        -> [Phase 2] expanded_queries populated
        -> [Phase 3] documents populated (retrieved + reranked, scored)
        -> [Phase 4] context populated (compressed from documents)
        -> [Phase 5] answer populated (or fallback triggered)
        -> [Eval]    eval_scores populated

All phases append to `trace` for observability.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(BaseModel):
    """
    A single chunk of text from a source document.

    Produced by Phase 1 (ingestion) and scored/ranked by Phase 3 (retrieval).
    score starts at 0.0 and is set by the retriever/reranker.
    chunk_index tracks position within the source for "Lost in the Middle" analysis.
    """

    id: str = Field(default_factory=_uuid)
    content: str
    source: str  # file path, URL, or logical identifier (e.g. "acme-q4-report.pdf")
    chunk_index: Optional[int] = None  # position of this chunk within source; None if not chunked
    metadata: dict[str, Any] = Field(default_factory=dict)  # arbitrary key/value from the parser
    score: float = 0.0  # relevance score assigned by retriever or reranker; higher = more relevant


class TraceEntry(BaseModel):
    """
    A single observability record written by one phase.

    Phases should write one TraceEntry per call, recording what they did and how
    long it took. The `details` dict is free-form — put phase-specific diagnostics
    here (e.g. number of queries expanded, reranker model name, tokens consumed).
    """

    phase: str  # e.g. "ingestion", "query_expansion", "retrieval", "reranking", "compression", "agent"
    timestamp: str = Field(default_factory=_now)
    duration_ms: float
    details: dict[str, Any] = Field(default_factory=dict)


class EvalScores(BaseModel):
    """
    Evaluation metrics populated by the evaluation layer.

    faithfulness:      [0, 1] — does the answer derive only from the retrieved context?
                       Measures hallucination. 1.0 = fully grounded, 0.0 = fully hallucinated.

    answer_relevance:  [0, 1] — is the answer relevant to the question?
                       Measures response quality. 1.0 = directly addresses the question.

    context_precision: [0, 1] — is the gold (relevant) document ranked highly?
                       Measures retrieval quality. 1.0 = gold doc is rank 1.
                       Requires a ground-truth reference to compute.

    All fields are Optional: they are None during normal inference runs and are
    only populated when the evaluation layer is explicitly invoked.
    """

    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None


class DataPacket(BaseModel):
    """
    The single object that flows through every phase of the pipeline.

    Design rules:
    - Phases NEVER mutate this object. Always: return packet.model_copy(update={...})
    - A field being empty ([] or "") means that phase has not run yet, not that it failed.
    - Add a TraceEntry for every phase that runs, even if output is empty.
    - session_id groups all packets from one user request for Langfuse tracing.
    """

    # ── Identity ────────────────────────────────────────────────────────────────
    session_id: str = Field(default_factory=_uuid)
    created_at: str = Field(default_factory=_now)

    # ── Phase 0: raw user input ──────────────────────────────────────────────────
    query: str  # the original, unmodified user query; never overwrite this

    # ── Phase 2: query intelligence output ──────────────────────────────────────
    # Populated by HyDE or Multi-Query Expansion.
    # Empty list means Phase 2 has not run; retrieval falls back to raw `query`.
    expanded_queries: list[str] = Field(default_factory=list)

    # ── Phase 3: retrieval + reranking output ────────────────────────────────────
    # Sorted descending by score after reranking. The retriever sets initial scores;
    # the reranker overwrites them. Top-k for LLM context is sliced by Phase 4.
    documents: list[Document] = Field(default_factory=list)

    # ── Phase 4: context engineering output ──────────────────────────────────────
    # The final compressed string passed to the LLM. Built from `documents` by
    # stripping low-similarity sentences. Empty string means Phase 4 has not run.
    context: str = ""

    # ── Phase 5: agentic reasoning output ────────────────────────────────────────
    # The final answer from the LLM. Empty string means Phase 5 has not run.
    # If C-RAG triggers a fallback, the agent sets answer = "" and adds a TraceEntry
    # with details={"fallback": "web_search"} or {"fallback": "refusal"}.
    answer: str = ""

    # ── Evaluation ───────────────────────────────────────────────────────────────
    eval_scores: EvalScores = Field(default_factory=EvalScores)

    # ── Observability ────────────────────────────────────────────────────────────
    trace: list[TraceEntry] = Field(default_factory=list)

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def with_trace(self, entry: TraceEntry) -> "DataPacket":
        """Return a copy of this packet with one trace entry appended."""
        return self.model_copy(update={"trace": self.trace + [entry]})

    def top_documents(self, k: int) -> list[Document]:
        """Return the top-k documents sorted by score descending."""
        return sorted(self.documents, key=lambda d: d.score, reverse=True)[:k]
