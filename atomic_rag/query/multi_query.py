"""
Multi-Query Expansion.

Problem: a single query phrasing may miss relevant documents that use different
vocabulary or framing.  A user who asks "auth flow" won't retrieve a chunk that
says "token validation pipeline" unless the embedder happens to align those terms.

Solution: generate N alternative phrasings of the original query, retrieve for
each independently, then merge results with Reciprocal Rank Fusion.  Different
phrasings hit different parts of the vocabulary distribution.

Reference: Ma et al., "Zero-Shot Listwise Document Reranking with a Large
Language Model" (2023) — discusses multi-query strategies in retrieval;
practical technique popularised by LangChain's MultiQueryRetriever.
"""

from __future__ import annotations

import re
import time

from atomic_rag.models.base import ChatModelBase
from atomic_rag.schema import DataPacket, TraceEntry

from .base import QueryExpansionBase

_DEFAULT_TEMPLATE = (
    "Generate {n} different search queries that would retrieve documents relevant "
    "to answering the question below. Each query should use different vocabulary or "
    "framing to maximise coverage. Output ONLY the queries, one per line, numbered "
    "like:\n1. ...\n2. ...\n\nQuestion: {query}"
)

_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)", re.MULTILINE)


def _parse_queries(text: str) -> list[str]:
    """Extract numbered lines from LLM output.  Falls back to line split."""
    matches = _LINE_RE.findall(text)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    # fallback: split on newlines, strip empties
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


class MultiQueryExpander(QueryExpansionBase):
    """
    Generates *n_queries* alternative phrasings of the user query and prepends
    the original.  HybridRetriever embeds each and fuses results via RRF.

    Parameters
    ----------
    chat_model:
        Any ChatModelBase implementation (Ollama, OpenAI, …).
    n_queries:
        Number of alternative queries to generate (not counting the original).
    prompt_template:
        Optional override.  Must contain ``{query}`` and ``{n}`` placeholders.
    include_original:
        If True (default), the original query is prepended to `expanded_queries`
        so that it is always retrieved against.
    fallback_to_original:
        If True (default), on LLM failure return only the original query.
    """

    def __init__(
        self,
        chat_model: ChatModelBase,
        n_queries: int = 3,
        prompt_template: str | None = None,
        include_original: bool = True,
        fallback_to_original: bool = True,
    ) -> None:
        self._chat = chat_model
        self._n = n_queries
        self._template = prompt_template or _DEFAULT_TEMPLATE
        self._include_original = include_original
        self._fallback = fallback_to_original

    def expand(self, packet: DataPacket) -> DataPacket:
        t0 = time.monotonic()
        error: str | None = None
        alternatives: list[str] = []

        try:
            prompt = self._template.format(n=self._n, query=packet.query)
            raw = self._chat.complete(prompt)
            alternatives = _parse_queries(raw)[: self._n]
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            if not self._fallback:
                raise

        if self._include_original:
            expanded = [packet.query] + alternatives
        else:
            expanded = alternatives or [packet.query]

        duration_ms = (time.monotonic() - t0) * 1000
        details: dict = {
            "strategy": "multi_query",
            "requested": self._n,
            "generated": len(alternatives),
            "expanded_count": len(expanded),
        }
        if error:
            details["error"] = error

        entry = TraceEntry(phase="query_expansion", duration_ms=duration_ms, details=details)
        return packet.model_copy(update={"expanded_queries": expanded}).with_trace(entry)
