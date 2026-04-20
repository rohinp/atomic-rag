"""
HyDE — Hypothetical Document Embeddings.

Problem: short user queries ("authentication flow?") live in a very different
embedding space from long, detailed document chunks.  Cosine similarity between
a 4-word query and a 200-word passage is low even when semantically aligned.

Solution: ask the LLM to write a *hypothetical* document that would answer the
query, then embed that document instead of the raw query.  The hypothetical doc
is long, uses domain vocabulary, and sits in roughly the same embedding space as
real retrieved passages — so cosine similarity scores are more discriminating.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (2022). https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

import time

from atomic_rag.models.base import ChatModelBase
from atomic_rag.schema import DataPacket, TraceEntry

from .base import QueryExpansionBase

_DEFAULT_TEMPLATE = (
    "Write a short passage (3–5 sentences) that directly answers the following question. "
    "Be specific and use domain terminology. Do not add a preamble or heading — "
    "start immediately with the answer text.\n\n"
    "Question: {query}"
)


class HyDEExpander(QueryExpansionBase):
    """
    Generates one hypothetical document per query and uses it as the expanded
    query for embedding-based retrieval.

    The original query is discarded from `expanded_queries` (the hypothetical
    doc already encodes its intent).  HybridRetriever will still have the
    original in `packet.query` for BM25 keyword search.

    Parameters
    ----------
    chat_model:
        Any ChatModelBase implementation (Ollama, OpenAI, …).
    prompt_template:
        Optional override.  Must contain ``{query}`` placeholder.
    fallback_to_original:
        If True, on LLM failure return the original query as the single
        expanded query rather than propagating the error.
    """

    def __init__(
        self,
        chat_model: ChatModelBase,
        prompt_template: str | None = None,
        fallback_to_original: bool = True,
    ) -> None:
        self._chat = chat_model
        self._template = prompt_template or _DEFAULT_TEMPLATE
        self._fallback = fallback_to_original

    def expand(self, packet: DataPacket) -> DataPacket:
        t0 = time.monotonic()
        error: str | None = None
        hypothetical: str | None = None

        try:
            prompt = self._template.format(query=packet.query)
            hypothetical = self._chat.complete(prompt).strip()
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            if not self._fallback:
                raise

        if hypothetical:
            expanded = [hypothetical]
        else:
            expanded = [packet.query]  # graceful fallback

        duration_ms = (time.monotonic() - t0) * 1000
        details: dict = {
            "strategy": "hyde",
            "expanded_count": len(expanded),
            "hypothetical_length": len(expanded[0]) if expanded else 0,
        }
        if error:
            details["error"] = error

        entry = TraceEntry(phase="query_expansion", duration_ms=duration_ms, details=details)
        return packet.model_copy(update={"expanded_queries": expanded}).with_trace(entry)
