"""
Tests for Phase 5 — Agent (C-RAG: Corrective Retrieval Augmented Generation).

All tests are offline: ChatModelBase is injected as a fake returning
predictable strings.  No LLM or network access required.
"""

from __future__ import annotations

import pytest

from atomic_rag.agent import (
    AgentRunner,
    EvaluatorBase,
    GeneratorBase,
    LLMEvaluator,
    LLMGenerator,
)
from atomic_rag.models.base import ChatModelBase
from atomic_rag.schema import DataPacket, Document, TraceEntry


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeChat(ChatModelBase):
    def __init__(self, response: str = "0.8") -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        return self._response

    def chat(self, messages: list[dict]) -> str:
        return self._response


class RecordingChat(ChatModelBase):
    """Captures every prompt it receives."""

    def __init__(self, response: str = "0.9") -> None:
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response

    def chat(self, messages: list[dict]) -> str:
        return self._response


class FakeEvaluator(EvaluatorBase):
    def __init__(self, score: float = 0.9) -> None:
        self._score = score

    def evaluate(self, query: str, context: str) -> float:
        return self._score


class FakeGenerator(GeneratorBase):
    def __init__(self, answer: str = "The answer is 42.") -> None:
        self._answer = answer

    def generate(self, query: str, context: str) -> str:
        return self._answer


def _packet(query: str = "What is X?", context: str = "X is a thing.") -> DataPacket:
    return DataPacket(query=query, context=context)


# ---------------------------------------------------------------------------
# EvaluatorBase contract
# ---------------------------------------------------------------------------


class TestEvaluatorBase:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            EvaluatorBase()  # type: ignore[abstract]

    def test_subclass_must_implement_evaluate(self):
        class Incomplete(EvaluatorBase):
            pass

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# GeneratorBase contract
# ---------------------------------------------------------------------------


class TestGeneratorBase:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            GeneratorBase()  # type: ignore[abstract]

    def test_subclass_must_implement_generate(self):
        class Incomplete(GeneratorBase):
            pass

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# LLMEvaluator
# ---------------------------------------------------------------------------


class TestLLMEvaluator:
    # --- Score parsing ---

    def test_parses_plain_float(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("0.8"))
        score = evaluator.evaluate("q", "context")
        assert abs(score - 0.8) < 1e-6

    def test_parses_1_0(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("1.0"))
        score = evaluator.evaluate("q", "context")
        assert score == 1.0

    def test_parses_0_0(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("0.0"))
        score = evaluator.evaluate("q", "context")
        assert score == 0.0

    def test_parses_float_embedded_in_text(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("Score: 0.7 out of 1.0"))
        score = evaluator.evaluate("q", "context")
        assert abs(score - 0.7) < 1e-6

    def test_parses_yes_as_1(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("YES"))
        score = evaluator.evaluate("q", "context")
        assert score == 1.0

    def test_parses_no_as_0(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("NO"))
        score = evaluator.evaluate("q", "context")
        assert score == 0.0

    def test_clamps_above_1(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("1.5"))
        score = evaluator.evaluate("q", "context")
        assert score == 1.0

    def test_clamps_below_0(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("-0.3"))
        score = evaluator.evaluate("q", "context")
        assert score == 0.0

    def test_default_score_on_unparseable(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("banana"), default_score=0.5)
        score = evaluator.evaluate("q", "context")
        assert score == 0.5

    def test_custom_default_score(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("banana"), default_score=0.3)
        score = evaluator.evaluate("q", "context")
        assert score == 0.3

    def test_empty_context_returns_zero_without_llm(self):
        chat = RecordingChat()
        evaluator = LLMEvaluator(chat_model=chat)
        score = evaluator.evaluate("q", "")
        assert score == 0.0
        assert len(chat.prompts) == 0  # LLM not called

    def test_whitespace_only_context_returns_zero(self):
        chat = RecordingChat()
        evaluator = LLMEvaluator(chat_model=chat)
        score = evaluator.evaluate("q", "   \n  ")
        assert score == 0.0

    # --- Prompt construction ---

    def test_prompt_includes_query(self):
        chat = RecordingChat()
        evaluator = LLMEvaluator(chat_model=chat)
        evaluator.evaluate("my specific question", "some context")
        assert "my specific question" in chat.prompts[0]

    def test_prompt_includes_context(self):
        chat = RecordingChat()
        evaluator = LLMEvaluator(chat_model=chat)
        evaluator.evaluate("q", "my specific context text")
        assert "my specific context text" in chat.prompts[0]

    def test_custom_template_used(self):
        chat = RecordingChat()
        evaluator = LLMEvaluator(
            chat_model=chat,
            prompt_template="Rate: {query} | {context}",
        )
        evaluator.evaluate("Q", "C")
        assert chat.prompts[0] == "Rate: Q | C"

    def test_returns_float(self):
        evaluator = LLMEvaluator(chat_model=FakeChat("0.6"))
        result = evaluator.evaluate("q", "ctx")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# LLMGenerator
# ---------------------------------------------------------------------------


class TestLLMGenerator:
    def test_returns_string(self):
        generator = LLMGenerator(chat_model=FakeChat("The answer."))
        result = generator.generate("q", "ctx")
        assert isinstance(result, str)

    def test_returns_chat_response(self):
        generator = LLMGenerator(chat_model=FakeChat("specific answer text"))
        result = generator.generate("q", "ctx")
        assert result == "specific answer text"

    def test_strips_whitespace(self):
        generator = LLMGenerator(chat_model=FakeChat("  padded answer  "))
        result = generator.generate("q", "ctx")
        assert result == "padded answer"

    def test_prompt_includes_query(self):
        chat = RecordingChat(response="ans")
        generator = LLMGenerator(chat_model=chat)
        generator.generate("unique question text", "ctx")
        assert "unique question text" in chat.prompts[0]

    def test_prompt_includes_context(self):
        chat = RecordingChat(response="ans")
        generator = LLMGenerator(chat_model=chat)
        generator.generate("q", "unique context material")
        assert "unique context material" in chat.prompts[0]

    def test_custom_template_used(self):
        chat = RecordingChat(response="ans")
        generator = LLMGenerator(
            chat_model=chat,
            prompt_template="Q={query} C={context}",
        )
        generator.generate("myQ", "myC")
        assert chat.prompts[0] == "Q=myQ C=myC"


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------


class TestAgentRunner:
    # --- Basic output ---

    def test_returns_datapacket(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        result = runner.run(_packet())
        assert isinstance(result, DataPacket)

    def test_populates_answer_on_high_score(self):
        runner = AgentRunner(
            evaluator=FakeEvaluator(0.9),
            generator=FakeGenerator("generated answer"),
        )
        result = runner.run(_packet())
        assert result.answer == "generated answer"

    def test_uses_fallback_on_low_score(self):
        runner = AgentRunner(
            evaluator=FakeEvaluator(0.1),
            generator=FakeGenerator("should not appear"),
            fallback_message="insufficient context",
        )
        result = runner.run(_packet())
        assert result.answer == "insufficient context"

    def test_threshold_boundary_exact_match_generates(self):
        runner = AgentRunner(
            evaluator=FakeEvaluator(0.5),
            generator=FakeGenerator("answer"),
            threshold=0.5,
        )
        result = runner.run(_packet())
        assert result.answer == "answer"

    def test_threshold_just_below_uses_fallback(self):
        runner = AgentRunner(
            evaluator=FakeEvaluator(0.499),
            generator=FakeGenerator("answer"),
            threshold=0.5,
        )
        result = runner.run(_packet())
        assert result.answer != "answer"

    def test_custom_fallback_message(self):
        runner = AgentRunner(
            evaluator=FakeEvaluator(0.0),
            generator=FakeGenerator(),
            fallback_message="custom fallback text",
        )
        result = runner.run(_packet())
        assert result.answer == "custom fallback text"

    # --- Immutability ---

    def test_does_not_mutate_input_packet(self):
        packet = _packet()
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        runner.run(packet)
        assert packet.answer == ""

    def test_original_query_preserved(self):
        packet = _packet(query="original question")
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        result = runner.run(packet)
        assert result.query == "original question"

    def test_context_preserved(self):
        packet = _packet(context="original context")
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        result = runner.run(packet)
        assert result.context == "original context"

    # --- Trace ---

    def test_appends_trace_entry(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        result = runner.run(_packet())
        assert len(result.trace) == 1

    def test_trace_phase_is_agent(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        result = runner.run(_packet())
        assert result.trace[-1].phase == "agent"

    def test_trace_records_eval_score(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.75), generator=FakeGenerator())
        result = runner.run(_packet())
        assert result.trace[-1].details["eval_score"] == 0.75

    def test_trace_records_threshold(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator(), threshold=0.6)
        result = runner.run(_packet())
        assert result.trace[-1].details["threshold"] == 0.6

    def test_trace_fallback_false_when_generated(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        result = runner.run(_packet())
        assert result.trace[-1].details["fallback"] is False

    def test_trace_fallback_true_when_triggered(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.1), generator=FakeGenerator())
        result = runner.run(_packet())
        assert result.trace[-1].details["fallback"] is True

    def test_trace_duration_positive(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        result = runner.run(_packet())
        assert result.trace[-1].duration_ms >= 0

    def test_trace_records_answer_length(self):
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator("hello"))
        result = runner.run(_packet())
        assert result.trace[-1].details["answer_length"] == 5

    def test_existing_trace_preserved(self):
        prior = TraceEntry(phase="retrieval", duration_ms=10.0)
        packet = _packet().with_trace(prior)
        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=FakeGenerator())
        result = runner.run(packet)
        assert len(result.trace) == 2
        assert result.trace[0].phase == "retrieval"

    # --- Evaluator receives correct inputs ---

    def test_evaluator_receives_query_and_context(self):
        received: list[tuple] = []

        class SpyEvaluator(EvaluatorBase):
            def evaluate(self, query: str, context: str) -> float:
                received.append((query, context))
                return 0.9

        runner = AgentRunner(evaluator=SpyEvaluator(), generator=FakeGenerator())
        runner.run(_packet(query="the Q", context="the C"))
        assert received == [("the Q", "the C")]

    # --- Generator receives correct inputs ---

    def test_generator_receives_query_and_context(self):
        received: list[tuple] = []

        class SpyGenerator(GeneratorBase):
            def generate(self, query: str, context: str) -> str:
                received.append((query, context))
                return "ans"

        runner = AgentRunner(evaluator=FakeEvaluator(0.9), generator=SpyGenerator())
        runner.run(_packet(query="the Q", context="the C"))
        assert received == [("the Q", "the C")]

    def test_generator_not_called_on_fallback(self):
        calls: list[str] = []

        class TrackingGenerator(GeneratorBase):
            def generate(self, query: str, context: str) -> str:
                calls.append("called")
                return "ans"

        runner = AgentRunner(evaluator=FakeEvaluator(0.0), generator=TrackingGenerator())
        runner.run(_packet())
        assert calls == []

    # --- Full pipeline integration ---

    def test_full_pipeline_packet_flow(self):
        """DataPacket flows through all 4 phases + agent correctly."""
        from atomic_rag.schema import Document

        packet = DataPacket(
            query="What is X?",
            context="[Source: test.py]\nX is a modular RAG framework.",
            documents=[
                Document(content="X is a modular RAG framework.", source="test.py")
            ],
        )
        runner = AgentRunner(
            evaluator=FakeEvaluator(0.95),
            generator=FakeGenerator("X is a modular RAG framework."),
        )
        result = runner.run(packet)

        assert result.answer == "X is a modular RAG framework."
        assert result.query == "What is X?"
        assert len(result.documents) == 1
        assert result.trace[-1].phase == "agent"
