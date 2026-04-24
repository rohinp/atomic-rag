"""
Tests for atomic_rag.retrieval — RRF, ChromaVectorStore, BM25Retriever,
CrossEncoderReranker, and HybridRetriever.

All run offline. External dependencies (chromadb, rank_bm25,
sentence_transformers) are mocked via injected fakes or monkeypatching.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from atomic_rag.retrieval.base import RetrieverBase
from atomic_rag.retrieval.bm25 import BM25Retriever
from atomic_rag.retrieval.fusion import reciprocal_rank_fusion
from atomic_rag.retrieval.hybrid import HybridRetriever
from atomic_rag.retrieval.reranker import CrossEncoderReranker
from atomic_rag.retrieval.vector_store import ChromaVectorStore
from atomic_rag.schema import DataPacket, Document, TraceEntry


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_doc(content: str, score: float = 0.0, idx: int = 0) -> Document:
    return Document(content=content, source="test.py", chunk_index=idx, score=score)


def make_packet(query: str = "test query") -> DataPacket:
    return DataPacket(query=query)


def make_embedder(vector: list[float] | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = vector or [0.1, 0.2, 0.3]
    embedder.embed_batch.side_effect = lambda texts: [[0.1] * 3 for _ in texts]
    return embedder


# ── RetrieverBase ─────────────────────────────────────────────────────────────

class TestRetrieverBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            RetrieverBase()

    def test_concrete_subclass_works(self):
        class Stub(RetrieverBase):
            def add_documents(self, docs): pass
            def retrieve(self, packet, top_k=5): return packet

        r = Stub()
        p = make_packet()
        assert r.retrieve(p) is p


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────

class TestRRF:
    def _docs(self, n: int) -> list[Document]:
        return [make_doc(f"doc {i}", idx=i) for i in range(n)]

    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []

    def test_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_passthrough(self):
        docs = self._docs(3)
        result = reciprocal_rank_fusion([docs])
        assert len(result) == 3
        assert [r.id for r in result] == [d.id for d in docs]

    def test_rank_1_beats_rank_2(self):
        docs = self._docs(2)
        # docs[0] is rank 0 (best), docs[1] is rank 1
        result = reciprocal_rank_fusion([[docs[0], docs[1]]])
        assert result[0].id == docs[0].id

    def test_doc_in_two_lists_outscores_doc_in_one(self):
        shared = make_doc("shared")
        unique = make_doc("unique")
        # shared appears in both lists at rank 0; unique only in list 2 at rank 0
        result = reciprocal_rank_fusion([[shared], [shared, unique]])
        assert result[0].id == shared.id

    def test_rrf_score_is_set_on_returned_docs(self):
        docs = self._docs(2)
        result = reciprocal_rank_fusion([[docs[0], docs[1]]])
        assert all(r.score > 0 for r in result)

    def test_deduplication_across_lists(self):
        doc = make_doc("dup")
        result = reciprocal_rank_fusion([[doc], [doc], [doc]])
        assert len(result) == 1

    def test_original_docs_not_mutated(self):
        docs = self._docs(2)
        original_scores = [d.score for d in docs]
        reciprocal_rank_fusion([[docs[0], docs[1]]])
        assert [d.score for d in docs] == original_scores

    def test_sorted_descending_by_score(self):
        docs = self._docs(5)
        result = reciprocal_rank_fusion([docs])
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_higher_k_reduces_score_gap(self):
        docs = self._docs(2)
        result_low_k = reciprocal_rank_fusion([docs], k=1)
        result_high_k = reciprocal_rank_fusion([docs], k=1000)
        gap_low = result_low_k[0].score - result_low_k[1].score
        gap_high = result_high_k[0].score - result_high_k[1].score
        assert gap_low > gap_high


# ── ChromaVectorStore ─────────────────────────────────────────────────────────

class TestChromaVectorStore:
    def _make_store(self) -> tuple[ChromaVectorStore, MagicMock]:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        store = ChromaVectorStore(_client=mock_client)
        return store, mock_collection

    def test_add_calls_collection_add(self):
        store, coll = self._make_store()
        doc = make_doc("hello")
        store.add([doc], [[0.1, 0.2]])
        coll.add.assert_called_once()

    def test_add_passes_ids_content_embeddings(self):
        store, coll = self._make_store()
        doc = make_doc("content here")
        store.add([doc], [[0.5, 0.6]])
        kwargs = coll.add.call_args.kwargs
        assert doc.id in kwargs["ids"]
        assert "content here" in kwargs["documents"]
        assert [0.5, 0.6] in kwargs["embeddings"]

    def test_add_empty_list_does_nothing(self):
        store, coll = self._make_store()
        store.add([], [])
        coll.add.assert_not_called()

    def test_search_returns_documents(self):
        store, coll = self._make_store()
        doc = make_doc("result doc")
        coll.query.return_value = {
            "ids": [[doc.id]],
            "documents": [[doc.content]],
            "metadatas": [[{"_source": "test.py", "_chunk_index": 0}]],
            "distances": [[0.1]],
        }
        results = store.search([0.1, 0.2], top_k=1)
        assert len(results) == 1
        assert results[0].content == "result doc"

    def test_search_score_is_one_minus_distance(self):
        store, coll = self._make_store()
        doc = make_doc("doc")
        coll.query.return_value = {
            "ids": [[doc.id]],
            "documents": [["doc"]],
            "metadatas": [[{"_source": "s.py"}]],
            "distances": [[0.2]],
        }
        results = store.search([0.1], top_k=1)
        assert abs(results[0].score - 0.8) < 1e-6

    def test_search_returns_empty_when_collection_empty(self):
        store, coll = self._make_store()
        coll.count.return_value = 0
        results = store.search([0.1], top_k=5)
        assert results == []
        coll.query.assert_not_called()

    def test_source_restored_on_returned_docs(self):
        store, coll = self._make_store()
        doc = make_doc("doc")
        coll.query.return_value = {
            "ids": [[doc.id]],
            "documents": [["doc"]],
            "metadatas": [[{"_source": "my_file.py", "_chunk_index": 3}]],
            "distances": [[0.0]],
        }
        results = store.search([0.1], top_k=1)
        assert results[0].source == "my_file.py"
        assert results[0].chunk_index == 3

    def test_import_error_when_chromadb_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "chromadb", None)
        store = ChromaVectorStore()  # no injected client
        with pytest.raises(ImportError, match="pip install chromadb"):
            store._make_client()

    def test_collection_created_with_cosine_space(self):
        store, _ = self._make_store()
        store._get_collection()  # triggers get_or_create
        call_kwargs = store._client.get_or_create_collection.call_args.kwargs
        assert call_kwargs.get("metadata", {}).get("hnsw:space") == "cosine"


# ── BM25Retriever ─────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def _make_retriever(self, docs: list[Document], monkeypatch) -> BM25Retriever:
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = MagicMock(
            tolist=lambda: [1.0] * len(docs)
        )
        mock_bm25_class = MagicMock(return_value=mock_bm25_instance)
        mock_rank_bm25 = MagicMock()
        mock_rank_bm25.BM25Okapi = mock_bm25_class
        monkeypatch.setitem(sys.modules, "rank_bm25", mock_rank_bm25)

        retriever = BM25Retriever()
        retriever.add(docs)
        return retriever

    def test_search_on_empty_returns_empty(self):
        r = BM25Retriever()
        assert r.search("anything", top_k=5) == []

    def test_import_error_when_rank_bm25_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rank_bm25", None)
        r = BM25Retriever()
        with pytest.raises(ImportError, match="pip install rank-bm25"):
            r.add([make_doc("test")])

    def test_add_builds_index(self, monkeypatch):
        docs = [make_doc(f"doc {i}") for i in range(3)]
        r = self._make_retriever(docs, monkeypatch)
        assert r._bm25 is not None
        assert len(r._docs) == 3

    def test_search_returns_documents(self, monkeypatch):
        docs = [make_doc("relevant content"), make_doc("other")]
        r = self._make_retriever(docs, monkeypatch)
        results = r.search("relevant", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(d, Document) for d in results)

    def test_search_filters_zero_score_docs(self, monkeypatch):
        docs = [make_doc("match"), make_doc("nomatch")]
        mock_instance = MagicMock()
        mock_instance.get_scores.return_value = MagicMock(tolist=lambda: [1.5, 0.0])
        mock_class = MagicMock(return_value=mock_instance)
        mock_mod = MagicMock()
        mock_mod.BM25Okapi = mock_class
        monkeypatch.setitem(sys.modules, "rank_bm25", mock_mod)

        r = BM25Retriever()
        r.add(docs)
        results = r.search("query", top_k=10)
        assert len(results) == 1
        assert results[0].content == "match"

    def test_search_scores_set_on_results(self, monkeypatch):
        docs = [make_doc("a"), make_doc("b")]
        mock_instance = MagicMock()
        mock_instance.get_scores.return_value = MagicMock(tolist=lambda: [2.5, 1.0])
        mock_class = MagicMock(return_value=mock_instance)
        mock_mod = MagicMock()
        mock_mod.BM25Okapi = mock_class
        monkeypatch.setitem(sys.modules, "rank_bm25", mock_mod)

        r = BM25Retriever()
        r.add(docs)
        results = r.search("q", top_k=2)
        assert results[0].score == 2.5
        assert results[1].score == 1.0

    def test_original_docs_not_mutated(self, monkeypatch):
        docs = [make_doc("original", score=0.0)]
        r = self._make_retriever(docs, monkeypatch)
        r.search("query", top_k=1)
        assert docs[0].score == 0.0

    def test_camelcase_identifier_matches_query(self):
        """BM25 must find 'DataPacket' when query contains 'datapacket'.

        With whitespace-only splitting 'DataPacket(BaseModel):' becomes one
        unmatched token. Alphanumeric tokenisation extracts 'datapacket'
        as a separate token, making it matchable.

        Note: BM25Okapi IDF formula is log(N-n+0.5) - log(n+0.5). With only
        2 documents and a term present in exactly 1, IDF = log(1.5)-log(1.5) = 0,
        which makes all scores 0. A corpus of 3+ documents is required for
        non-zero IDF when a term appears in exactly one document.
        """
        from rank_bm25 import BM25Okapi  # real library required

        relevant = make_doc("class DataPacket(BaseModel):\n    '''The inter-module contract.'''")
        noise1 = make_doc("def ingest(self, path): pass")
        noise2 = make_doc("class HybridRetriever combines vector and keyword search")

        r = BM25Retriever()
        r.add([relevant, noise1, noise2])
        results = r.search("how does DataPacket work?", top_k=5)

        assert len(results) >= 1
        assert results[0].content == relevant.content

    def test_punctuation_in_query_ignored(self):
        """Punctuation in the query should not prevent matching.

        Note: BM25Okapi needs N >= 3 with term in 1 doc for non-zero IDF.
        """
        from rank_bm25 import BM25Okapi  # real library required

        doc = make_doc("retrieval augmented generation pipeline")
        noise1 = make_doc("class DataPacket inter module contract")
        noise2 = make_doc("def ingest path source file")
        r = BM25Retriever()
        r.add([doc, noise1, noise2])
        results = r.search("retrieval-augmented generation?", top_k=5)
        assert len(results) >= 1
        assert results[0].content == doc.content


# ── CrossEncoderReranker ──────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    def _make_reranker(self, scores: list[float]) -> CrossEncoderReranker:
        mock_model = MagicMock()
        mock_model.predict.return_value = MagicMock(tolist=lambda: scores)
        return CrossEncoderReranker(_model_instance=mock_model)

    def test_rerank_returns_top_k(self):
        docs = [make_doc(f"doc {i}") for i in range(5)]
        r = self._make_reranker([0.9, 0.3, 0.7, 0.1, 0.5])
        results = r.rerank("query", docs, top_k=3)
        assert len(results) == 3

    def test_rerank_sorts_descending(self):
        docs = [make_doc(f"doc {i}") for i in range(3)]
        r = self._make_reranker([0.2, 0.9, 0.5])
        results = r.rerank("query", docs, top_k=3)
        assert results[0].score == 0.9
        assert results[1].score == 0.5
        assert results[2].score == 0.2

    def test_rerank_scores_written_to_docs(self):
        docs = [make_doc("a", score=0.0), make_doc("b", score=0.0)]
        r = self._make_reranker([0.8, 0.4])
        results = r.rerank("q", docs, top_k=2)
        result_scores = sorted([res.score for res in results], reverse=True)
        assert result_scores == [0.8, 0.4]

    def test_rerank_empty_docs_returns_empty(self):
        r = self._make_reranker([])
        assert r.rerank("q", [], top_k=5) == []

    def test_original_docs_not_mutated(self):
        docs = [make_doc("doc", score=0.1)]
        r = self._make_reranker([0.9])
        r.rerank("q", docs, top_k=1)
        assert docs[0].score == 0.1

    def test_import_error_when_sentence_transformers_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        r = CrossEncoderReranker()  # no injected model
        with pytest.raises(ImportError, match="pip install sentence-transformers"):
            r._get_model()

    def test_predict_called_with_query_doc_pairs(self):
        docs = [make_doc("alpha"), make_doc("beta")]
        mock_model = MagicMock()
        mock_model.predict.return_value = MagicMock(tolist=lambda: [0.5, 0.7])
        r = CrossEncoderReranker(_model_instance=mock_model)
        r.rerank("my query", docs, top_k=2)
        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0] == ("my query", "alpha")
        assert pairs[1] == ("my query", "beta")


# ── HybridRetriever ───────────────────────────────────────────────────────────

class TestHybridRetriever:
    def _make_retriever(
        self,
        vector_docs: list[Document] | None = None,
        bm25_docs: list[Document] | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> HybridRetriever:
        embedder = make_embedder()

        mock_store = MagicMock(spec=ChromaVectorStore)
        mock_store.search.return_value = vector_docs or []
        mock_store.add.return_value = None

        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=mock_store,
            reranker=reranker,
        )
        # Patch internal BM25 retriever
        mock_bm25 = MagicMock(spec=BM25Retriever)
        mock_bm25.search.return_value = bm25_docs or []
        retriever._bm25 = mock_bm25

        return retriever

    # ── retrieve contract ─────────────────────────────────────────────────────

    def test_retrieve_returns_datapacket(self):
        r = self._make_retriever()
        result = r.retrieve(make_packet())
        assert isinstance(result, DataPacket)

    def test_retrieve_does_not_mutate_input_packet(self):
        r = self._make_retriever()
        packet = make_packet()
        r.retrieve(packet)
        assert packet.documents == []
        assert packet.trace == []

    def test_retrieve_populates_documents(self):
        docs = [make_doc("result", score=0.9)]
        r = self._make_retriever(vector_docs=docs)
        result = r.retrieve(make_packet())
        assert len(result.documents) > 0

    def test_retrieve_adds_trace_entry(self):
        r = self._make_retriever()
        result = r.retrieve(make_packet())
        assert len(result.trace) == 1
        assert result.trace[0].phase == "retrieval"

    def test_trace_entry_has_expected_details(self):
        docs = [make_doc("d")]
        r = self._make_retriever(vector_docs=docs, bm25_docs=docs)
        result = r.retrieve(make_packet())
        details = result.trace[0].details
        assert "vector_hits" in details
        assert "bm25_hits" in details
        assert "top_k" in details
        assert "reranked" in details

    def test_retrieve_preserves_existing_trace(self):
        r = self._make_retriever()
        prior = TraceEntry(phase="ingestion", duration_ms=5.0)
        packet = make_packet()
        packet = packet.with_trace(prior)
        result = r.retrieve(packet)
        assert len(result.trace) == 2
        assert result.trace[0].phase == "ingestion"

    # ── top_k ─────────────────────────────────────────────────────────────────

    def test_retrieve_respects_top_k(self):
        docs = [make_doc(f"doc {i}", score=float(i)) for i in range(10)]
        r = self._make_retriever(vector_docs=docs)
        result = r.retrieve(make_packet(), top_k=3)
        assert len(result.documents) <= 3

    # ── both sources called ───────────────────────────────────────────────────

    def test_both_vector_and_bm25_are_queried(self):
        r = self._make_retriever()
        r.retrieve(make_packet("hello"))
        r.vector_store.search.assert_called_once()
        r._bm25.search.assert_called_once()

    def test_embedder_called_with_query(self):
        r = self._make_retriever()
        r.retrieve(make_packet("my query"))
        r.embedder.embed.assert_called_once_with("my query")

    # ── reranker integration ──────────────────────────────────────────────────

    def test_reranker_called_when_provided(self):
        docs = [make_doc("doc")]
        mock_reranker = MagicMock(spec=CrossEncoderReranker)
        mock_reranker.rerank.return_value = docs
        r = self._make_retriever(vector_docs=docs, reranker=mock_reranker)
        r.retrieve(make_packet(), top_k=1)
        mock_reranker.rerank.assert_called_once()

    def test_reranker_not_called_when_absent(self):
        docs = [make_doc("doc")]
        r = self._make_retriever(vector_docs=docs, reranker=None)
        result = r.retrieve(make_packet(), top_k=5)
        assert "reranked" in result.trace[0].details
        assert result.trace[0].details["reranked"] is False

    def test_trace_records_reranked_true(self):
        docs = [make_doc("doc")]
        mock_reranker = MagicMock(spec=CrossEncoderReranker)
        mock_reranker.rerank.return_value = docs
        r = self._make_retriever(vector_docs=docs, reranker=mock_reranker)
        result = r.retrieve(make_packet())
        assert result.trace[0].details["reranked"] is True

    # ── add_documents ─────────────────────────────────────────────────────────

    def test_add_documents_calls_embed_batch(self):
        r = self._make_retriever()
        docs = [make_doc("a"), make_doc("b")]
        r.add_documents(docs)
        r.embedder.embed_batch.assert_called_once_with(["a", "b"])

    def test_add_documents_calls_vector_store_add(self):
        r = self._make_retriever()
        docs = [make_doc("a")]
        r.add_documents(docs)
        r.vector_store.add.assert_called_once()

    def test_add_documents_calls_bm25_add(self):
        r = self._make_retriever()
        docs = [make_doc("a")]
        r.add_documents(docs)
        r._bm25.add.assert_called_once_with(docs)

    def test_add_empty_docs_does_nothing(self):
        r = self._make_retriever()
        r.add_documents([])
        r.embedder.embed_batch.assert_not_called()
        r.vector_store.add.assert_not_called()

    def test_bm25_only_and_vector_only_docs_both_surface_in_hybrid_results(self):
        """Demonstrates the complementarity of hybrid search (Cormack et al., 2009).

        A document found exclusively by BM25 (exact keyword match, not in vector
        results) and a document found exclusively by vector search (semantic match,
        not in BM25 results) both appear in the final fused ranking. Neither
        retrieval modality alone would surface both documents — only hybrid
        fusion via RRF guarantees coverage across both signals.

        This addresses Failure Point 4 from Barnett et al. (2024):
        "The correct documents are not retrieved."

        References:
          Cormack et al., "Reciprocal Rank Fusion" (2009) — https://dl.acm.org/doi/10.1145/1571941.1572114
          Barnett et al., "Seven Failure Points" (2024) — https://arxiv.org/abs/2401.05856
        """
        bm25_only = make_doc("bm25_exclusive_result")
        vector_only = make_doc("vector_exclusive_result")

        r = self._make_retriever(vector_docs=[vector_only], bm25_docs=[bm25_only])
        result = r.retrieve(make_packet("query"), top_k=10)

        content_set = {d.content for d in result.documents}
        assert "vector_exclusive_result" in content_set, "Vector-only doc missing from hybrid results"
        assert "bm25_exclusive_result" in content_set, "BM25-only doc missing from hybrid results"
