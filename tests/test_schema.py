"""
Tests for atomic_rag.schema — the inter-module DataPacket contract.

These tests verify:
- Default values and field invariants
- Immutability pattern (model_copy, not mutation)
- Serialisation round-trip (JSON in == JSON out)
- Helper methods
- That required fields are enforced
"""

import json
import time

import pytest
from pydantic import ValidationError

from atomic_rag.schema import DataPacket, Document, EvalScores, TraceEntry


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_packet() -> DataPacket:
    return DataPacket(query="What is the capital of France?")


@pytest.fixture
def sample_document() -> Document:
    return Document(content="Paris is the capital of France.", source="geography.pdf")


@pytest.fixture
def sample_trace_entry() -> TraceEntry:
    return TraceEntry(phase="retrieval", duration_ms=42.5, details={"top_k": 5})


# ── Document ──────────────────────────────────────────────────────────────────

class TestDocument:
    def test_required_fields_enforced(self):
        with pytest.raises(ValidationError):
            Document()  # content and source are required

    def test_id_auto_generated(self, sample_document):
        assert sample_document.id != ""
        assert len(sample_document.id) == 36  # UUID4 format

    def test_two_documents_have_different_ids(self):
        d1 = Document(content="a", source="x.pdf")
        d2 = Document(content="a", source="x.pdf")
        assert d1.id != d2.id

    def test_score_defaults_to_zero(self, sample_document):
        assert sample_document.score == 0.0

    def test_chunk_index_defaults_to_none(self, sample_document):
        assert sample_document.chunk_index is None

    def test_metadata_defaults_to_empty_dict(self, sample_document):
        assert sample_document.metadata == {}

    def test_custom_fields_accepted(self):
        doc = Document(
            content="text",
            source="report.pdf",
            chunk_index=3,
            score=0.87,
            metadata={"page": 12, "language": "en"},
        )
        assert doc.chunk_index == 3
        assert doc.score == 0.87
        assert doc.metadata["page"] == 12

    def test_explicit_id_is_preserved(self):
        doc = Document(id="my-custom-id", content="text", source="a.pdf")
        assert doc.id == "my-custom-id"


# ── TraceEntry ────────────────────────────────────────────────────────────────

class TestTraceEntry:
    def test_required_fields_enforced(self):
        with pytest.raises(ValidationError):
            TraceEntry()  # phase and duration_ms are required

    def test_timestamp_auto_generated(self, sample_trace_entry):
        assert sample_trace_entry.timestamp != ""
        # ISO 8601 format check
        assert "T" in sample_trace_entry.timestamp

    def test_details_defaults_to_empty_dict(self):
        entry = TraceEntry(phase="ingestion", duration_ms=10.0)
        assert entry.details == {}

    def test_details_accepts_arbitrary_values(self, sample_trace_entry):
        assert sample_trace_entry.details["top_k"] == 5


# ── EvalScores ────────────────────────────────────────────────────────────────

class TestEvalScores:
    def test_both_scores_default_to_none(self):
        scores = EvalScores()
        assert scores.faithfulness is None
        assert scores.context_precision is None

    def test_scores_can_be_set(self):
        scores = EvalScores(faithfulness=0.95, context_precision=0.80)
        assert scores.faithfulness == 0.95
        assert scores.context_precision == 0.80


# ── DataPacket ────────────────────────────────────────────────────────────────

class TestDataPacket:
    def test_query_is_required(self):
        with pytest.raises(ValidationError):
            DataPacket()

    def test_minimal_packet_defaults(self, minimal_packet):
        assert minimal_packet.query == "What is the capital of France?"
        assert minimal_packet.expanded_queries == []
        assert minimal_packet.documents == []
        assert minimal_packet.context == ""
        assert minimal_packet.answer == ""
        assert minimal_packet.trace == []
        assert minimal_packet.eval_scores.faithfulness is None

    def test_session_id_auto_generated(self, minimal_packet):
        assert len(minimal_packet.session_id) == 36

    def test_two_packets_have_different_session_ids(self):
        p1 = DataPacket(query="q")
        p2 = DataPacket(query="q")
        assert p1.session_id != p2.session_id

    def test_created_at_is_iso_format(self, minimal_packet):
        assert "T" in minimal_packet.created_at

    # ── Immutability pattern ──────────────────────────────────────────────────

    def test_model_copy_does_not_mutate_original(self, minimal_packet):
        updated = minimal_packet.model_copy(update={"answer": "Paris"})
        assert minimal_packet.answer == ""  # original unchanged
        assert updated.answer == "Paris"

    def test_model_copy_preserves_other_fields(self, minimal_packet):
        updated = minimal_packet.model_copy(update={"answer": "Paris"})
        assert updated.query == minimal_packet.query
        assert updated.session_id == minimal_packet.session_id

    # ── with_trace helper ─────────────────────────────────────────────────────

    def test_with_trace_appends_entry(self, minimal_packet, sample_trace_entry):
        updated = minimal_packet.with_trace(sample_trace_entry)
        assert len(updated.trace) == 1
        assert updated.trace[0].phase == "retrieval"

    def test_with_trace_does_not_mutate_original(self, minimal_packet, sample_trace_entry):
        minimal_packet.with_trace(sample_trace_entry)
        assert len(minimal_packet.trace) == 0

    def test_with_trace_is_chainable(self, minimal_packet):
        e1 = TraceEntry(phase="ingestion", duration_ms=10.0)
        e2 = TraceEntry(phase="retrieval", duration_ms=20.0)
        updated = minimal_packet.with_trace(e1).with_trace(e2)
        assert len(updated.trace) == 2
        assert updated.trace[0].phase == "ingestion"
        assert updated.trace[1].phase == "retrieval"

    # ── top_documents helper ──────────────────────────────────────────────────

    def test_top_documents_returns_sorted_by_score(self):
        docs = [
            Document(content="c", source="s.pdf", score=0.5),
            Document(content="a", source="s.pdf", score=0.9),
            Document(content="b", source="s.pdf", score=0.7),
        ]
        packet = DataPacket(query="q", documents=docs)
        top = packet.top_documents(k=2)
        assert len(top) == 2
        assert top[0].score == 0.9
        assert top[1].score == 0.7

    def test_top_documents_k_larger_than_available(self):
        docs = [Document(content="a", source="s.pdf", score=0.5)]
        packet = DataPacket(query="q", documents=docs)
        assert len(packet.top_documents(k=10)) == 1

    def test_top_documents_empty_list(self, minimal_packet):
        assert minimal_packet.top_documents(k=5) == []

    def test_top_documents_does_not_mutate_original_order(self):
        docs = [
            Document(content="low", source="s.pdf", score=0.1),
            Document(content="high", source="s.pdf", score=0.9),
        ]
        packet = DataPacket(query="q", documents=docs)
        packet.top_documents(k=2)
        assert packet.documents[0].score == 0.1  # original order preserved

    # ── Serialisation ─────────────────────────────────────────────────────────

    def test_json_round_trip(self, minimal_packet):
        as_json = minimal_packet.model_dump_json()
        restored = DataPacket.model_validate_json(as_json)
        assert restored.query == minimal_packet.query
        assert restored.session_id == minimal_packet.session_id

    def test_json_round_trip_with_documents_and_trace(self):
        doc = Document(content="Paris is the capital.", source="geo.pdf", score=0.9)
        entry = TraceEntry(phase="retrieval", duration_ms=15.0, details={"model": "bge"})
        packet = DataPacket(
            query="What is the capital?",
            documents=[doc],
            context="Paris is the capital.",
            answer="Paris",
            trace=[entry],
            eval_scores=EvalScores(faithfulness=1.0, context_precision=0.9),
        )
        restored = DataPacket.model_validate_json(packet.model_dump_json())
        assert restored.answer == "Paris"
        assert restored.documents[0].score == 0.9
        assert restored.trace[0].phase == "retrieval"
        assert restored.eval_scores.faithfulness == 1.0

    def test_model_dump_produces_valid_json(self, minimal_packet):
        dumped = minimal_packet.model_dump()
        # Should be serialisable to a JSON string without errors
        json.dumps(dumped)
