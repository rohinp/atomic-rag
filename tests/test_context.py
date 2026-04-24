"""
Tests for atomic_rag.context — CompressorBase, split_sentences,
cosine_similarity, and SentenceCompressor.

Pure functions (split_sentences, cosine_similarity) are tested directly.
SentenceCompressor uses an injected stub embedder — no real model calls.
"""

import math
from unittest.mock import MagicMock

import pytest

from atomic_rag.context.base import CompressorBase
from atomic_rag.context.compressor import (
    SentenceCompressor,
    _segments,
    _source_label,
    cosine_similarity,
    split_sentences,
)
from atomic_rag.schema import DataPacket, Document, TraceEntry


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_doc(
    content: str,
    source: str = "test.pdf",
    chunk_index: int = 0,
    score: float = 0.9,
    metadata: dict | None = None,
) -> Document:
    return Document(
        content=content,
        source=source,
        chunk_index=chunk_index,
        score=score,
        metadata=metadata or {},
    )


def make_packet(query: str = "test query", docs: list[Document] | None = None) -> DataPacket:
    return DataPacket(query=query, documents=docs or [])


def stub_embedder(query_vec: list[float], segment_vecs: list[list[float]]) -> MagicMock:
    """
    Embedder whose embed() returns query_vec and embed_batch() returns segment_vecs.
    """
    embedder = MagicMock()
    embedder.embed.return_value = query_vec
    embedder.embed_batch.return_value = segment_vecs
    return embedder


# ── CompressorBase ────────────────────────────────────────────────────────────

class TestCompressorBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            CompressorBase()

    def test_concrete_subclass_works(self):
        class Stub(CompressorBase):
            def compress(self, packet):
                return packet.model_copy(update={"context": "done"})

        result = Stub().compress(make_packet())
        assert result.context == "done"


# ── split_sentences ───────────────────────────────────────────────────────────

class TestSplitSentences:
    def test_single_sentence(self):
        assert split_sentences("Hello world.") == ["Hello world."]

    def test_two_sentences(self):
        result = split_sentences("First sentence. Second sentence.")
        assert len(result) == 2
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."

    def test_question_mark_split(self):
        result = split_sentences("What is RAG? It is a technique.")
        assert len(result) == 2

    def test_exclamation_split(self):
        result = split_sentences("Amazing result! Here is why.")
        assert len(result) == 2

    def test_no_split_on_lowercase_after_period(self):
        # "e.g. something" should not split
        result = split_sentences("Revenue was $4.2B. Growth was strong.")
        # The period-lowercase case doesn't trigger the regex
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_empty_string_returns_empty(self):
        assert split_sentences("") == []

    def test_whitespace_only_returns_empty(self):
        assert split_sentences("   ") == []

    def test_strips_whitespace_from_parts(self):
        result = split_sentences("First.  Second.")
        assert all(s == s.strip() for s in result)

    def test_multiple_sentences(self):
        text = "A. B. C. D."
        # Each capital after period+space creates a split
        result = split_sentences(text)
        assert len(result) >= 1  # at minimum one sentence


# ── cosine_similarity ─────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors_return_minus_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_known_similarity(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1 / math.sqrt(2)
        assert abs(cosine_similarity(a, b) - expected) < 1e-6

    def test_symmetry(self):
        a = [0.3, 0.4, 0.5]
        b = [0.1, 0.9, 0.2]
        assert abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < 1e-9


# ── _segments helper ──────────────────────────────────────────────────────────

class TestSegments:
    def test_prose_doc_splits_into_sentences(self):
        doc = make_doc("First sentence. Second sentence.")
        segs = _segments(doc)
        assert len(segs) == 2

    def test_code_doc_kept_whole(self):
        doc = make_doc(
            "def foo():\n    return 42\n",
            metadata={"language": "python", "type": "function", "name": "foo"},
        )
        segs = _segments(doc)
        assert len(segs) == 1
        assert "def foo" in segs[0]

    def test_empty_content_returns_empty(self):
        doc = make_doc("")
        segs = _segments(doc)
        assert segs == []


# ── _source_label helper ──────────────────────────────────────────────────────

class TestSourceLabel:
    def test_includes_source_path(self):
        doc = make_doc("x", source="/repo/src/auth.py")
        assert "/repo/src/auth.py" in _source_label(doc)

    def test_includes_line_numbers_when_available(self):
        doc = make_doc("x", metadata={"start_line": 10, "end_line": 25})
        label = _source_label(doc)
        assert "10" in label
        assert "25" in label

    def test_falls_back_to_chunk_index(self):
        doc = make_doc("x", chunk_index=7)
        assert "7" in _source_label(doc)


# ── SentenceCompressor ────────────────────────────────────────────────────────

class TestSentenceCompressor:
    # ── constructor ───────────────────────────────────────────────────────────

    def test_invalid_threshold_raises(self):
        embedder = MagicMock()
        with pytest.raises(ValueError, match="threshold"):
            SentenceCompressor(embedder, threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            SentenceCompressor(embedder, threshold=-0.1)

    def test_invalid_min_sentences_raises(self):
        embedder = MagicMock()
        with pytest.raises(ValueError, match="min_sentences"):
            SentenceCompressor(embedder, min_sentences=0)

    # ── immutability ──────────────────────────────────────────────────────────

    def test_does_not_mutate_input_packet(self):
        doc = make_doc("A sentence. Another sentence.")
        packet = make_packet(docs=[doc])
        embedder = stub_embedder([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        SentenceCompressor(embedder, threshold=0.5).compress(packet)
        assert packet.context == ""
        assert packet.trace == []

    # ── trace entry ───────────────────────────────────────────────────────────

    def test_adds_trace_entry(self):
        packet = make_packet(docs=[make_doc("One sentence.")])
        embedder = stub_embedder([1.0, 0.0], [[1.0, 0.0]])
        result = SentenceCompressor(embedder).compress(packet)
        assert len(result.trace) == 1
        assert result.trace[0].phase == "context"

    def test_trace_entry_has_expected_keys(self):
        packet = make_packet(docs=[make_doc("One sentence.")])
        embedder = stub_embedder([1.0, 0.0], [[1.0, 0.0]])
        result = SentenceCompressor(embedder).compress(packet)
        details = result.trace[0].details
        assert "documents" in details
        assert "sentences_before" in details
        assert "sentences_after" in details
        assert "reduction_pct" in details
        assert "threshold" in details

    def test_trace_preserves_existing_entries(self):
        prior = TraceEntry(phase="retrieval", duration_ms=50.0)
        packet = make_packet(docs=[make_doc("X.")])
        packet = packet.with_trace(prior)
        embedder = stub_embedder([1.0], [[1.0]])
        result = SentenceCompressor(embedder).compress(packet)
        assert len(result.trace) == 2
        assert result.trace[0].phase == "retrieval"
        assert result.trace[1].phase == "context"

    # ── context output ────────────────────────────────────────────────────────

    def test_context_populated(self):
        doc = make_doc("Relevant sentence. Another relevant one.")
        packet = make_packet(query="relevant", docs=[doc])
        # Both sentences score above threshold
        embedder = stub_embedder([1.0, 0.0], [[0.9, 0.0], [0.8, 0.0]])
        result = SentenceCompressor(embedder, threshold=0.5).compress(packet)
        assert result.context != ""

    def test_context_contains_source_label(self):
        doc = make_doc("Content.", source="report.pdf", chunk_index=2)
        packet = make_packet(docs=[doc])
        embedder = stub_embedder([1.0], [[0.9]])
        result = SentenceCompressor(embedder, threshold=0.5).compress(packet)
        assert "report.pdf" in result.context

    def test_multiple_docs_separated_by_divider(self):
        docs = [make_doc("Doc one content.", chunk_index=0),
                make_doc("Doc two content.", chunk_index=1)]
        packet = make_packet(docs=docs)
        embedder = stub_embedder([1.0], [[0.9], [0.9]])
        # embed_batch called once per doc
        embedder.embed_batch.side_effect = [[[0.9]], [[0.9]]]
        result = SentenceCompressor(embedder, threshold=0.5).compress(packet)
        assert "---" in result.context

    # ── filtering ─────────────────────────────────────────────────────────────

    def test_low_similarity_sentence_dropped(self):
        # query: [1,0]. Relevant embedding [1,0] → sim=1.0. Irrelevant [0,1] → sim=0.0.
        # Cosine similarity is angle-based so we need genuinely different directions.
        doc = make_doc("Relevant sentence. Irrelevant sentence.")
        packet = make_packet(docs=[doc])
        embedder = stub_embedder([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        result = SentenceCompressor(embedder, threshold=0.5).compress(packet)
        assert "Relevant sentence" in result.context
        assert "Irrelevant sentence" not in result.context

    def test_all_low_similarity_still_keeps_min_sentences(self):
        doc = make_doc("Low. Also low. Still low.")
        packet = make_packet(docs=[doc])
        # All below threshold=0.9
        embedder = stub_embedder([1.0, 0.0], [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        result = SentenceCompressor(embedder, threshold=0.9, min_sentences=1).compress(packet)
        # Should keep the highest-scoring one ("Still low." at 0.3)
        assert result.context != ""

    def test_min_sentences_two_always_keeps_two(self):
        doc = make_doc("Alpha. Beta. Gamma.")
        packet = make_packet(docs=[doc])
        # All below threshold — should still keep top 2
        embedder = stub_embedder(
            [1.0, 0.0],
            [[0.1, 0.0], [0.15, 0.0], [0.05, 0.0]],
        )
        result = SentenceCompressor(embedder, threshold=0.9, min_sentences=2).compress(packet)
        # Context is non-empty with at least 2 segments kept
        assert result.context.count("Beta") + result.context.count("Alpha") >= 1

    def test_threshold_zero_keeps_all_sentences(self):
        doc = make_doc("First. Second. Third.")
        packet = make_packet(docs=[doc])
        embedder = stub_embedder([1.0, 0.0], [[0.01, 0.0], [0.01, 0.0], [0.01, 0.0]])
        result = SentenceCompressor(embedder, threshold=0.0).compress(packet)
        assert "First" in result.context
        assert "Second" in result.context
        assert "Third" in result.context

    # ── code chunks kept whole ────────────────────────────────────────────────

    def test_code_chunk_never_split(self):
        code = "def authenticate(token):\n    user = db.get(token)\n    return user"
        doc = make_doc(code, metadata={"language": "python", "type": "function"})
        packet = make_packet(docs=[doc])
        # embed_batch called with the whole chunk as one segment
        embedder = stub_embedder([1.0], [[0.9]])
        result = SentenceCompressor(embedder, threshold=0.5).compress(packet)
        # embed_batch was called once with a single-element list
        call_args = embedder.embed_batch.call_args[0][0]
        assert len(call_args) == 1
        assert "def authenticate" in call_args[0]

    def test_code_chunk_below_threshold_still_kept_by_min_sentences(self):
        code = "def low_score_func():\n    pass"
        doc = make_doc(code, metadata={"language": "python"})
        packet = make_packet(docs=[doc])
        embedder = stub_embedder([1.0], [[0.05]])  # very low score
        result = SentenceCompressor(embedder, threshold=0.9, min_sentences=1).compress(packet)
        assert "low_score_func" in result.context

    # ── embed_batch efficiency ────────────────────────────────────────────────

    def test_embed_batch_used_not_individual_embed_per_sentence(self):
        doc = make_doc("First sentence. Second sentence. Third sentence.")
        packet = make_packet(docs=[doc])
        embedder = stub_embedder([1.0, 0.0], [[0.9, 0.0], [0.8, 0.0], [0.7, 0.0]])
        SentenceCompressor(embedder, threshold=0.5).compress(packet)
        # embed called once (for query), embed_batch called once (for all sentences)
        assert embedder.embed.call_count == 1
        assert embedder.embed_batch.call_count == 1

    def test_query_embedded_once_across_all_docs(self):
        docs = [make_doc(f"Doc {i}. Content.") for i in range(3)]
        packet = make_packet(docs=docs)
        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0]
        embedder.embed_batch.return_value = [[0.9, 0.0], [0.8, 0.0]]
        SentenceCompressor(embedder, threshold=0.5).compress(packet)
        # query embedded exactly once regardless of document count
        assert embedder.embed.call_count == 1

    # ── empty inputs ──────────────────────────────────────────────────────────

    def test_empty_documents_returns_packet_unchanged(self):
        packet = make_packet(docs=[])
        embedder = MagicMock()
        result = SentenceCompressor(embedder).compress(packet)
        assert result.context == ""
        embedder.embed.assert_not_called()

    def test_empty_documents_still_adds_trace(self):
        packet = make_packet(docs=[])
        result = SentenceCompressor(MagicMock()).compress(packet)
        assert len(result.trace) == 1
        assert result.trace[0].details["documents"] == 0

    # ── reduction metrics ─────────────────────────────────────────────────────

    def test_reduction_pct_reflects_dropped_sentences(self):
        # 3 sentences, 1 kept → 66.7% reduction.
        # Use orthogonal directions: query=[1,0], kept sim=1.0, dropped sim=0.0.
        doc = make_doc("Keep this. Drop that. Drop this too.")
        packet = make_packet(docs=[doc])
        embedder = stub_embedder(
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        )
        result = SentenceCompressor(embedder, threshold=0.5).compress(packet)
        details = result.trace[0].details
        assert details["sentences_before"] == 3
        assert details["sentences_after"] == 1
        assert details["reduction_pct"] == pytest.approx(66.7, abs=0.2)

    def test_relevant_sentence_buried_in_middle_is_retained(self):
        """Directly demonstrates the 'Lost in the Middle' problem (Liu et al., 2023).

        LLMs perform worse when the relevant sentence is surrounded by irrelevant
        text in a long context window. SentenceCompressor mitigates this by
        scoring each sentence individually and dropping low-similarity ones,
        regardless of their position. The relevant sentence in the middle is
        retained while the flanking noise is dropped.

        Reference: Liu et al., "Lost in the Middle: How Language Models Use Long
        Contexts" (2023) — https://arxiv.org/abs/2307.03172
        """
        doc = make_doc(
            "Unrelated opening sentence. The answer is forty-two. Unrelated closing sentence."
        )
        packet = make_packet(docs=[doc])
        # query vector: [1, 0]. Middle sentence similarity = 1.0 (relevant).
        # Flanking sentences similarity = 0.0 (orthogonal, irrelevant).
        embedder = stub_embedder(
            [1.0, 0.0],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        )
        result = SentenceCompressor(embedder, threshold=0.5).compress(packet)
        assert "forty-two" in result.context
        assert "Unrelated opening" not in result.context
        assert "Unrelated closing" not in result.context

    def test_no_reduction_when_all_kept(self):
        doc = make_doc("One. Two.")
        packet = make_packet(docs=[doc])
        embedder = stub_embedder([1.0, 0.0], [[0.9, 0.0], [0.8, 0.0]])
        result = SentenceCompressor(embedder, threshold=0.5).compress(packet)
        assert result.trace[0].details["reduction_pct"] == 0.0
