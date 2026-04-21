"""
Tests for the evaluation module (faithfulness, answer relevance, Ragas wrapper).

All tests are offline: ChatModelBase and EmbedderBase are injected as fakes.
The RagasEvaluator tests mock the ragas library to avoid the install requirement.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from atomic_rag.evaluation import (
    EmbeddingAnswerRelevance,
    LLMFaithfulnessScorer,
    PipelineEvalBase,
    RagasEvaluator,
)
from atomic_rag.models.base import ChatModelBase, EmbedderBase
from atomic_rag.schema import DataPacket, Document, EvalScores, TraceEntry


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeChat(ChatModelBase):
    def __init__(self, responses: list[str] | str = "YES") -> None:
        if isinstance(responses, str):
            self._responses = [responses]
        else:
            self._responses = list(responses)
        self._idx = 0

    def complete(self, prompt: str) -> str:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    def chat(self, messages: list[dict]) -> str:
        return self.complete("")


class FakeEmbedder(EmbedderBase):
    """Returns fixed vectors for query and answer, optionally different."""

    def __init__(self, query_vec: list[float] | None = None, answer_vec: list[float] | None = None) -> None:
        self._q = query_vec or [1.0, 0.0, 0.0]
        self._a = answer_vec or [1.0, 0.0, 0.0]
        self._call_count = 0

    def embed(self, text: str) -> list[float]:
        return self._q

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._call_count += 1
        return [self._q, self._a][: len(texts)]


def _packet(
    query: str = "What is X?",
    answer: str = "X is a modular RAG framework.",
    context: str = "X is a modular RAG framework that solves seven failure points.",
) -> DataPacket:
    return DataPacket(query=query, answer=answer, context=context)


# ---------------------------------------------------------------------------
# PipelineEvalBase contract
# ---------------------------------------------------------------------------


class TestPipelineEvalBase:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            PipelineEvalBase()  # type: ignore[abstract]

    def test_subclass_must_implement_score(self):
        class Incomplete(PipelineEvalBase):
            pass

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# EvalScores schema
# ---------------------------------------------------------------------------


class TestEvalScores:
    def test_all_fields_default_to_none(self):
        scores = EvalScores()
        assert scores.faithfulness is None
        assert scores.answer_relevance is None
        assert scores.context_precision is None

    def test_fields_can_be_set(self):
        scores = EvalScores(faithfulness=0.8, answer_relevance=0.9, context_precision=0.7)
        assert scores.faithfulness == 0.8
        assert scores.answer_relevance == 0.9
        assert scores.context_precision == 0.7

    def test_partial_update_preserves_other_fields(self):
        scores = EvalScores(faithfulness=0.8, answer_relevance=0.9)
        updated = scores.model_copy(update={"context_precision": 0.7})
        assert updated.faithfulness == 0.8
        assert updated.answer_relevance == 0.9
        assert updated.context_precision == 0.7


# ---------------------------------------------------------------------------
# LLMFaithfulnessScorer
# ---------------------------------------------------------------------------


class TestLLMFaithfulnessScorer:
    # --- Basic output ---

    def test_returns_datapacket(self):
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A\nClaim B", "YES", "YES"])
        )
        result = scorer.score(_packet())
        assert isinstance(result, DataPacket)

    def test_populates_faithfulness_score(self):
        # 2 claims, both YES → score = 1.0
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A\nClaim B", "YES", "YES"])
        )
        result = scorer.score(_packet())
        assert result.eval_scores.faithfulness == 1.0

    def test_partial_support_gives_fractional_score(self):
        # 2 claims, 1 YES 1 NO → score = 0.5
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A\nClaim B", "YES", "NO"])
        )
        result = scorer.score(_packet())
        assert result.eval_scores.faithfulness == 0.5

    def test_all_no_gives_zero(self):
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A\nClaim B", "NO", "NO"])
        )
        result = scorer.score(_packet())
        assert result.eval_scores.faithfulness == 0.0

    def test_empty_answer_gives_zero_without_llm(self):
        chat = FakeChat()
        scorer = LLMFaithfulnessScorer(chat_model=chat)
        result = scorer.score(_packet(answer=""))
        assert result.eval_scores.faithfulness == 0.0
        assert chat._idx == 0  # LLM not called

    def test_empty_context_gives_zero_without_llm(self):
        chat = FakeChat()
        scorer = LLMFaithfulnessScorer(chat_model=chat)
        result = scorer.score(_packet(context=""))
        assert result.eval_scores.faithfulness == 0.0
        assert chat._idx == 0

    def test_no_claims_extracted_gives_zero(self):
        # LLM returns empty string for claim extraction
        scorer = LLMFaithfulnessScorer(chat_model=FakeChat(""))
        result = scorer.score(_packet())
        assert result.eval_scores.faithfulness == 0.0

    # --- Claim parsing ---

    def test_strips_bullet_points_from_claims(self):
        recorded: list[str] = []

        class RecordingChat(ChatModelBase):
            call = 0

            def complete(self, prompt: str) -> str:
                RecordingChat.call += 1
                if RecordingChat.call == 1:
                    return "- Claim one\n- Claim two"
                recorded.append(prompt)
                return "YES"

            def chat(self, messages: list[dict]) -> str:
                return self.complete("")

        scorer = LLMFaithfulnessScorer(chat_model=RecordingChat())
        scorer.score(_packet())
        assert any("Claim one" in p for p in recorded)
        assert any("Claim two" in p for p in recorded)

    def test_strips_numbered_list_from_claims(self):
        recorded: list[str] = []

        class RecordingChat(ChatModelBase):
            call = 0

            def complete(self, prompt: str) -> str:
                RecordingChat.call += 1
                if RecordingChat.call == 1:
                    return "1. First claim\n2. Second claim"
                recorded.append(prompt)
                return "YES"

            def chat(self, messages: list[dict]) -> str:
                return self.complete("")

        scorer = LLMFaithfulnessScorer(chat_model=RecordingChat())
        scorer.score(_packet())
        assert any("First claim" in p for p in recorded)

    def test_case_insensitive_yes_detection(self):
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "yes"])
        )
        result = scorer.score(_packet())
        assert result.eval_scores.faithfulness == 1.0

    # --- Immutability ---

    def test_does_not_mutate_input_packet(self):
        packet = _packet()
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "YES"])
        )
        scorer.score(packet)
        assert packet.eval_scores.faithfulness is None

    def test_preserves_other_eval_scores(self):
        packet = _packet()
        packet = packet.model_copy(
            update={"eval_scores": EvalScores(answer_relevance=0.9)}
        )
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "YES"])
        )
        result = scorer.score(packet)
        assert result.eval_scores.answer_relevance == 0.9
        assert result.eval_scores.faithfulness is not None

    # --- Trace ---

    def test_appends_trace_entry(self):
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "YES"])
        )
        result = scorer.score(_packet())
        assert len(result.trace) == 1

    def test_trace_phase_is_evaluation(self):
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "YES"])
        )
        result = scorer.score(_packet())
        assert result.trace[-1].phase == "evaluation"

    def test_trace_metric_is_faithfulness(self):
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "YES"])
        )
        result = scorer.score(_packet())
        assert result.trace[-1].details["metric"] == "faithfulness"

    def test_trace_records_claim_counts(self):
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A\nClaim B", "YES", "NO"])
        )
        result = scorer.score(_packet())
        assert result.trace[-1].details["claims_extracted"] == 2
        assert result.trace[-1].details["claims_supported"] == 1

    def test_trace_duration_positive(self):
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "YES"])
        )
        result = scorer.score(_packet())
        assert result.trace[-1].duration_ms >= 0

    def test_existing_trace_entries_preserved(self):
        prior = TraceEntry(phase="retrieval", duration_ms=10.0)
        packet = _packet().with_trace(prior)
        scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "YES"])
        )
        result = scorer.score(packet)
        assert len(result.trace) == 2
        assert result.trace[0].phase == "retrieval"

    # --- Custom templates ---

    def test_custom_claim_template_is_used(self):
        prompts: list[str] = []

        class RecordingChat(ChatModelBase):
            call = 0

            def complete(self, prompt: str) -> str:
                prompts.append(prompt)
                RecordingChat.call += 1
                return "Claim A" if RecordingChat.call == 1 else "YES"

            def chat(self, messages: list[dict]) -> str:
                return self.complete("")

        scorer = LLMFaithfulnessScorer(
            chat_model=RecordingChat(),
            claim_template="Extract: {answer}",
        )
        scorer.score(_packet(answer="my answer"))
        assert prompts[0] == "Extract: my answer"

    def test_custom_verify_template_is_used(self):
        prompts: list[str] = []

        class RecordingChat(ChatModelBase):
            call = 0

            def complete(self, prompt: str) -> str:
                prompts.append(prompt)
                RecordingChat.call += 1
                return "Claim A" if RecordingChat.call == 1 else "YES"

            def chat(self, messages: list[dict]) -> str:
                return self.complete("")

        scorer = LLMFaithfulnessScorer(
            chat_model=RecordingChat(),
            verify_template="Verify: {claim} using {context}",
        )
        scorer.score(_packet(context="ctx"))
        assert prompts[1] == "Verify: Claim A using ctx"


# ---------------------------------------------------------------------------
# EmbeddingAnswerRelevance
# ---------------------------------------------------------------------------


class TestEmbeddingAnswerRelevance:
    def test_returns_datapacket(self):
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        result = scorer.score(_packet())
        assert isinstance(result, DataPacket)

    def test_identical_vectors_give_score_1(self):
        vec = [1.0, 0.0, 0.0]
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder(query_vec=vec, answer_vec=vec))
        result = scorer.score(_packet())
        assert abs(result.eval_scores.answer_relevance - 1.0) < 1e-6

    def test_orthogonal_vectors_give_score_0(self):
        scorer = EmbeddingAnswerRelevance(
            embedder=FakeEmbedder(query_vec=[1.0, 0.0], answer_vec=[0.0, 1.0])
        )
        result = scorer.score(_packet())
        assert abs(result.eval_scores.answer_relevance) < 1e-6

    def test_opposite_vectors_give_score_0(self):
        # cosine of 180° = -1 → clamped to 0.0
        scorer = EmbeddingAnswerRelevance(
            embedder=FakeEmbedder(query_vec=[1.0, 0.0], answer_vec=[-1.0, 0.0])
        )
        result = scorer.score(_packet())
        assert result.eval_scores.answer_relevance == 0.0

    def test_empty_answer_gives_zero_without_embed(self):
        embedder = FakeEmbedder()
        scorer = EmbeddingAnswerRelevance(embedder=embedder)
        result = scorer.score(_packet(answer=""))
        assert result.eval_scores.answer_relevance == 0.0
        assert embedder._call_count == 0

    def test_empty_query_gives_zero(self):
        embedder = FakeEmbedder()
        scorer = EmbeddingAnswerRelevance(embedder=embedder)
        result = scorer.score(_packet(query=""))
        assert result.eval_scores.answer_relevance == 0.0

    def test_uses_embed_batch(self):
        embedder = FakeEmbedder()
        scorer = EmbeddingAnswerRelevance(embedder=embedder)
        scorer.score(_packet())
        assert embedder._call_count == 1  # single batch call, not two separate embeds

    def test_does_not_mutate_input_packet(self):
        packet = _packet()
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        scorer.score(packet)
        assert packet.eval_scores.answer_relevance is None

    def test_preserves_other_eval_scores(self):
        packet = _packet().model_copy(
            update={"eval_scores": EvalScores(faithfulness=0.8)}
        )
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        result = scorer.score(packet)
        assert result.eval_scores.faithfulness == 0.8
        assert result.eval_scores.answer_relevance is not None

    def test_appends_trace_entry(self):
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        result = scorer.score(_packet())
        assert len(result.trace) == 1

    def test_trace_phase_is_evaluation(self):
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        result = scorer.score(_packet())
        assert result.trace[-1].phase == "evaluation"

    def test_trace_metric_is_answer_relevance(self):
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        result = scorer.score(_packet())
        assert result.trace[-1].details["metric"] == "answer_relevance"

    def test_trace_records_score(self):
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        result = scorer.score(_packet())
        assert "score" in result.trace[-1].details

    def test_trace_duration_positive(self):
        scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        result = scorer.score(_packet())
        assert result.trace[-1].duration_ms >= 0


# ---------------------------------------------------------------------------
# Chaining evaluators
# ---------------------------------------------------------------------------


class TestEvaluatorChaining:
    def test_two_evaluators_populate_separate_fields(self):
        packet = _packet()

        faith_scorer = LLMFaithfulnessScorer(
            chat_model=FakeChat(["Claim A", "YES"])
        )
        rel_scorer = EmbeddingAnswerRelevance(
            embedder=FakeEmbedder([1.0, 0.0], [1.0, 0.0])
        )

        packet = faith_scorer.score(packet)
        packet = rel_scorer.score(packet)

        assert packet.eval_scores.faithfulness == 1.0
        assert packet.eval_scores.answer_relevance == 1.0
        assert len(packet.trace) == 2

    def test_second_evaluator_does_not_overwrite_first(self):
        packet = _packet().model_copy(
            update={"eval_scores": EvalScores(faithfulness=0.75)}
        )
        rel_scorer = EmbeddingAnswerRelevance(embedder=FakeEmbedder())
        result = rel_scorer.score(packet)

        assert result.eval_scores.faithfulness == 0.75
        assert result.eval_scores.answer_relevance is not None


# ---------------------------------------------------------------------------
# RagasEvaluator (mocked)
# ---------------------------------------------------------------------------


class TestRagasEvaluator:
    def _make_mock_ragas(self, scores: dict[str, float]):
        """Return a mock ragas module and result DataFrame."""
        import pandas as pd

        mock_result = MagicMock()
        mock_result.to_pandas.return_value = pd.DataFrame([scores])

        mock_ragas = MagicMock()
        mock_ragas.evaluate.return_value = mock_result
        mock_ragas.SingleTurnSample = MagicMock(return_value=MagicMock())
        mock_ragas.EvaluationDataset = MagicMock(return_value=MagicMock())

        return mock_ragas

    def test_maps_faithfulness_to_eval_scores(self):
        mock_ragas = self._make_mock_ragas({"faithfulness": 0.85})
        metric = MagicMock()
        metric.__class__.__name__ = "Faithfulness"

        with patch.dict(sys.modules, {"ragas": mock_ragas, "ragas.metrics": MagicMock()}):
            evaluator = RagasEvaluator(metrics=[metric])
            result = evaluator.score(_packet())

        assert result.eval_scores.faithfulness == 0.85

    def test_maps_answer_relevancy_to_answer_relevance(self):
        mock_ragas = self._make_mock_ragas({"answer_relevancy": 0.92})
        metric = MagicMock()
        metric.__class__.__name__ = "AnswerRelevancy"

        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[metric])
            result = evaluator.score(_packet())

        assert result.eval_scores.answer_relevance == 0.92

    def test_maps_context_precision(self):
        mock_ragas = self._make_mock_ragas({"context_precision": 0.78})
        metric = MagicMock()
        metric.__class__.__name__ = "ContextPrecision"

        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[metric])
            result = evaluator.score(_packet())

        assert result.eval_scores.context_precision == 0.78

    def test_multiple_metrics_populate_multiple_fields(self):
        mock_ragas = self._make_mock_ragas({
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
        })
        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock(), MagicMock()])
            result = evaluator.score(_packet())

        assert result.eval_scores.faithfulness == 0.8
        assert result.eval_scores.answer_relevance == 0.9

    def test_uses_document_contents_as_contexts(self):
        """retrieved_contexts should come from packet.documents when present."""
        mock_ragas = self._make_mock_ragas({"faithfulness": 0.9})

        captured: list = []

        def capture_sample(**kwargs):
            captured.append(kwargs)
            return MagicMock()

        mock_ragas.SingleTurnSample.side_effect = capture_sample

        docs = [
            Document(content="Doc A content", source="a.py"),
            Document(content="Doc B content", source="b.py"),
        ]
        packet = _packet()
        packet = packet.model_copy(update={"documents": docs})

        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock()])
            evaluator.score(packet)

        assert captured[0]["retrieved_contexts"] == ["Doc A content", "Doc B content"]

    def test_falls_back_to_context_string_when_no_documents(self):
        mock_ragas = self._make_mock_ragas({"faithfulness": 0.9})
        captured: list = []

        def capture_sample(**kwargs):
            captured.append(kwargs)
            return MagicMock()

        mock_ragas.SingleTurnSample.side_effect = capture_sample

        packet = _packet(context="fallback context text")

        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock()])
            evaluator.score(packet)

        assert captured[0]["retrieved_contexts"] == ["fallback context text"]

    def test_passes_reference_when_provided(self):
        mock_ragas = self._make_mock_ragas({"context_precision": 0.7})
        captured: list = []

        def capture_sample(**kwargs):
            captured.append(kwargs)
            return MagicMock()

        mock_ragas.SingleTurnSample.side_effect = capture_sample

        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock()], reference="ground truth")
            evaluator.score(_packet())

        assert captured[0]["reference"] == "ground truth"

    def test_no_reference_field_when_not_provided(self):
        mock_ragas = self._make_mock_ragas({"faithfulness": 0.8})
        captured: list = []

        def capture_sample(**kwargs):
            captured.append(kwargs)
            return MagicMock()

        mock_ragas.SingleTurnSample.side_effect = capture_sample

        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock()])
            evaluator.score(_packet())

        assert "reference" not in captured[0]

    def test_appends_trace_entry(self):
        mock_ragas = self._make_mock_ragas({"faithfulness": 0.8})
        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock()])
            result = evaluator.score(_packet())
        assert len(result.trace) == 1

    def test_trace_phase_is_evaluation(self):
        mock_ragas = self._make_mock_ragas({"faithfulness": 0.8})
        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock()])
            result = evaluator.score(_packet())
        assert result.trace[-1].phase == "evaluation"

    def test_trace_evaluator_is_ragas(self):
        mock_ragas = self._make_mock_ragas({"faithfulness": 0.8})
        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock()])
            result = evaluator.score(_packet())
        assert result.trace[-1].details["evaluator"] == "ragas"

    def test_import_error_when_ragas_not_installed(self):
        with patch.dict(sys.modules, {"ragas": None}):
            evaluator = RagasEvaluator(metrics=[])
            with pytest.raises(ImportError, match="ragas"):
                evaluator.score(_packet())

    def test_does_not_mutate_input_packet(self):
        mock_ragas = self._make_mock_ragas({"faithfulness": 0.8})
        packet = _packet()
        with patch.dict(sys.modules, {"ragas": mock_ragas}):
            evaluator = RagasEvaluator(metrics=[MagicMock()])
            evaluator.score(packet)
        assert packet.eval_scores.faithfulness is None
