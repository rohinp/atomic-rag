"""
Tests for Phase 2 — Query Intelligence (HyDE + Multi-Query Expansion).

All tests are offline: the ChatModelBase is injected as a simple fake that
returns a predictable string, so no LLM is needed.
"""

from __future__ import annotations

import pytest

from atomic_rag.models.base import ChatModelBase
from atomic_rag.query import HyDEExpander, MultiQueryExpander, QueryExpansionBase
from atomic_rag.schema import DataPacket


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeChat(ChatModelBase):
    """Returns a fixed string for every complete() call."""

    def __init__(self, response: str = "This is a hypothetical answer.") -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        return self._response

    def chat(self, messages: list[dict]) -> str:
        return self._response


class RaisingChat(ChatModelBase):
    """Always raises RuntimeError — used to test fallback behaviour."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("LLM unreachable")

    def chat(self, messages: list[dict]) -> str:
        raise RuntimeError("LLM unreachable")


# ---------------------------------------------------------------------------
# QueryExpansionBase contract
# ---------------------------------------------------------------------------


class TestQueryExpansionBase:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            QueryExpansionBase()  # type: ignore[abstract]

    def test_subclass_must_implement_expand(self):
        class Incomplete(QueryExpansionBase):
            pass

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# HyDEExpander
# ---------------------------------------------------------------------------


class TestHyDEExpander:
    def _packet(self, query: str = "How does authentication work?") -> DataPacket:
        return DataPacket(query=query)

    # --- Basic functionality ---

    def test_returns_datapacket(self):
        expander = HyDEExpander(chat_model=FakeChat())
        result = expander.expand(self._packet())
        assert isinstance(result, DataPacket)

    def test_populates_expanded_queries(self):
        expander = HyDEExpander(chat_model=FakeChat("Hypothetical document text."))
        result = expander.expand(self._packet())
        assert result.expanded_queries == ["Hypothetical document text."]

    def test_expanded_queries_has_exactly_one_entry(self):
        expander = HyDEExpander(chat_model=FakeChat("Some hypothetical text"))
        result = expander.expand(self._packet())
        assert len(result.expanded_queries) == 1

    def test_strips_whitespace_from_response(self):
        expander = HyDEExpander(chat_model=FakeChat("  Answer with spaces.  "))
        result = expander.expand(self._packet())
        assert result.expanded_queries[0] == "Answer with spaces."

    def test_original_query_preserved(self):
        packet = self._packet("original query")
        expander = HyDEExpander(chat_model=FakeChat("hypothetical"))
        result = expander.expand(packet)
        assert result.query == "original query"

    def test_does_not_mutate_input_packet(self):
        packet = self._packet()
        expander = HyDEExpander(chat_model=FakeChat("hypothetical"))
        expander.expand(packet)
        assert packet.expanded_queries == []

    # --- Trace ---

    def test_appends_trace_entry(self):
        expander = HyDEExpander(chat_model=FakeChat())
        result = expander.expand(self._packet())
        assert len(result.trace) == 1

    def test_trace_phase_is_query_expansion(self):
        expander = HyDEExpander(chat_model=FakeChat())
        result = expander.expand(self._packet())
        assert result.trace[-1].phase == "query_expansion"

    def test_trace_strategy_is_hyde(self):
        expander = HyDEExpander(chat_model=FakeChat())
        result = expander.expand(self._packet())
        assert result.trace[-1].details["strategy"] == "hyde"

    def test_trace_duration_is_positive(self):
        expander = HyDEExpander(chat_model=FakeChat())
        result = expander.expand(self._packet())
        assert result.trace[-1].duration_ms >= 0

    def test_trace_records_hypothetical_length(self):
        expander = HyDEExpander(chat_model=FakeChat("Hello world!"))
        result = expander.expand(self._packet())
        assert result.trace[-1].details["hypothetical_length"] == len("Hello world!")

    def test_trace_records_expanded_count(self):
        expander = HyDEExpander(chat_model=FakeChat("text"))
        result = expander.expand(self._packet())
        assert result.trace[-1].details["expanded_count"] == 1

    def test_existing_trace_preserved(self):
        from atomic_rag.schema import TraceEntry

        packet = self._packet()
        prior = TraceEntry(phase="ingestion", duration_ms=10.0)
        packet = packet.with_trace(prior)
        expander = HyDEExpander(chat_model=FakeChat())
        result = expander.expand(packet)
        assert len(result.trace) == 2
        assert result.trace[0].phase == "ingestion"

    # --- Fallback behaviour ---

    def test_fallback_to_original_on_error(self):
        expander = HyDEExpander(chat_model=RaisingChat(), fallback_to_original=True)
        result = expander.expand(self._packet("my query"))
        assert result.expanded_queries == ["my query"]

    def test_fallback_records_error_in_trace(self):
        expander = HyDEExpander(chat_model=RaisingChat(), fallback_to_original=True)
        result = expander.expand(self._packet())
        assert "error" in result.trace[-1].details

    def test_no_fallback_propagates_error(self):
        expander = HyDEExpander(chat_model=RaisingChat(), fallback_to_original=False)
        with pytest.raises(RuntimeError):
            expander.expand(self._packet())

    # --- Custom prompt template ---

    def test_custom_template_is_used(self):
        calls: list[str] = []

        class RecordingChat(ChatModelBase):
            def complete(self, prompt: str) -> str:
                calls.append(prompt)
                return "response"

            def chat(self, messages: list[dict]) -> str:
                return "response"

        template = "Custom: {query}"
        expander = HyDEExpander(chat_model=RecordingChat(), prompt_template=template)
        expander.expand(self._packet("test"))
        assert calls[0] == "Custom: test"


# ---------------------------------------------------------------------------
# MultiQueryExpander
# ---------------------------------------------------------------------------

_MULTI_RESPONSE = "1. How does auth work?\n2. What is token validation?\n3. Explain the login flow."


class TestMultiQueryExpander:
    def _packet(self, query: str = "How does authentication work?") -> DataPacket:
        return DataPacket(query=query)

    # --- Basic functionality ---

    def test_returns_datapacket(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        result = expander.expand(self._packet())
        assert isinstance(result, DataPacket)

    def test_expanded_queries_non_empty(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        result = expander.expand(self._packet())
        assert len(result.expanded_queries) > 0

    def test_original_included_by_default(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        result = expander.expand(self._packet("my query"))
        assert result.expanded_queries[0] == "my query"

    def test_alternatives_appended_after_original(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE), n_queries=3)
        result = expander.expand(self._packet("my query"))
        assert len(result.expanded_queries) > 1
        # alternatives should contain parsed lines
        alts = result.expanded_queries[1:]
        assert any("auth" in a.lower() or "token" in a.lower() or "login" in a.lower() for a in alts)

    def test_n_queries_limits_alternatives(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE), n_queries=2)
        result = expander.expand(self._packet())
        # original + 2 alternatives = 3 total
        assert len(result.expanded_queries) == 3

    def test_original_not_included_when_disabled(self):
        expander = MultiQueryExpander(
            chat_model=FakeChat(_MULTI_RESPONSE),
            n_queries=3,
            include_original=False,
        )
        result = expander.expand(self._packet("my query"))
        assert "my query" not in result.expanded_queries

    def test_does_not_mutate_input_packet(self):
        packet = self._packet()
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        expander.expand(packet)
        assert packet.expanded_queries == []

    def test_original_query_preserved(self):
        packet = self._packet("original query")
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        result = expander.expand(packet)
        assert result.query == "original query"

    # --- Parsing ---

    def test_parses_numbered_lines(self):
        expander = MultiQueryExpander(
            chat_model=FakeChat("1. First query\n2. Second query"),
            n_queries=2,
            include_original=False,
        )
        result = expander.expand(self._packet())
        assert "First query" in result.expanded_queries
        assert "Second query" in result.expanded_queries

    def test_parses_parenthesis_numbered_lines(self):
        expander = MultiQueryExpander(
            chat_model=FakeChat("1) First query\n2) Second query"),
            n_queries=2,
            include_original=False,
        )
        result = expander.expand(self._packet())
        assert "First query" in result.expanded_queries

    def test_falls_back_to_line_split_without_numbers(self):
        expander = MultiQueryExpander(
            chat_model=FakeChat("alpha query\nbeta query"),
            n_queries=2,
            include_original=False,
        )
        result = expander.expand(self._packet())
        assert "alpha query" in result.expanded_queries

    # --- Trace ---

    def test_appends_trace_entry(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        result = expander.expand(self._packet())
        assert len(result.trace) == 1

    def test_trace_phase_is_query_expansion(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        result = expander.expand(self._packet())
        assert result.trace[-1].phase == "query_expansion"

    def test_trace_strategy_is_multi_query(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        result = expander.expand(self._packet())
        assert result.trace[-1].details["strategy"] == "multi_query"

    def test_trace_records_requested_and_generated(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE), n_queries=3)
        result = expander.expand(self._packet())
        assert result.trace[-1].details["requested"] == 3
        assert result.trace[-1].details["generated"] == 3

    def test_trace_duration_is_positive(self):
        expander = MultiQueryExpander(chat_model=FakeChat(_MULTI_RESPONSE))
        result = expander.expand(self._packet())
        assert result.trace[-1].duration_ms >= 0

    # --- Fallback behaviour ---

    def test_fallback_returns_original_on_error(self):
        expander = MultiQueryExpander(chat_model=RaisingChat(), fallback_to_original=True)
        result = expander.expand(self._packet("my query"))
        assert result.expanded_queries == ["my query"]

    def test_fallback_records_error_in_trace(self):
        expander = MultiQueryExpander(chat_model=RaisingChat(), fallback_to_original=True)
        result = expander.expand(self._packet())
        assert "error" in result.trace[-1].details

    def test_no_fallback_propagates_error(self):
        expander = MultiQueryExpander(chat_model=RaisingChat(), fallback_to_original=False)
        with pytest.raises(RuntimeError):
            expander.expand(self._packet())

    # --- Custom template ---

    def test_custom_template_receives_n_and_query(self):
        calls: list[str] = []

        class RecordingChat(ChatModelBase):
            def complete(self, prompt: str) -> str:
                calls.append(prompt)
                return "1. q1\n2. q2"

            def chat(self, messages: list[dict]) -> str:
                return "1. q1\n2. q2"

        template = "Gen {n} queries for: {query}"
        expander = MultiQueryExpander(
            chat_model=RecordingChat(),
            n_queries=2,
            prompt_template=template,
        )
        expander.expand(self._packet("test question"))
        assert calls[0] == "Gen 2 queries for: test question"


# ---------------------------------------------------------------------------
# HybridRetriever integration with expanded_queries
# ---------------------------------------------------------------------------


class TestHybridRetrieverWithExpandedQueries:
    """
    Verify that HybridRetriever uses expanded_queries for vector search
    when they are populated.  Uses a mock embedder and vector store.
    """

    def _make_retriever(self, embed_calls: list[str]):
        from atomic_rag.models.base import EmbedderBase
        from atomic_rag.retrieval.bm25 import BM25Retriever
        from atomic_rag.retrieval.hybrid import HybridRetriever
        from atomic_rag.retrieval.vector_store import ChromaVectorStore

        class TrackingEmbedder(EmbedderBase):
            def embed(self, text: str) -> list[float]:
                embed_calls.append(text)
                return [0.1, 0.2, 0.3]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        return HybridRetriever(embedder=TrackingEmbedder())

    def test_single_embed_call_without_expanded_queries(self):
        calls: list[str] = []
        retriever = self._make_retriever(calls)
        packet = DataPacket(query="original query")
        retriever.retrieve(packet, top_k=3)
        # Only the original query is embedded for vector search
        assert calls == ["original query"]

    def test_multiple_embed_calls_with_expanded_queries(self):
        calls: list[str] = []
        retriever = self._make_retriever(calls)
        packet = DataPacket(query="original", expanded_queries=["alt1", "alt2"])
        retriever.retrieve(packet, top_k=3)
        # Vector search uses expanded_queries, not original
        assert calls == ["alt1", "alt2"]

    def test_trace_records_queries_used_count(self):
        calls: list[str] = []
        retriever = self._make_retriever(calls)
        packet = DataPacket(query="q", expanded_queries=["e1", "e2", "e3"])
        result = retriever.retrieve(packet, top_k=2)
        trace = next(t for t in result.trace if t.phase == "retrieval")
        assert trace.details["queries_used"] == 3

    def test_trace_queries_used_is_one_without_expansion(self):
        calls: list[str] = []
        retriever = self._make_retriever(calls)
        packet = DataPacket(query="q")
        result = retriever.retrieve(packet, top_k=2)
        trace = next(t for t in result.trace if t.phase == "retrieval")
        assert trace.details["queries_used"] == 1
