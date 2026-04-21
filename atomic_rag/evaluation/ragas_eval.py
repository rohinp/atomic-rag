"""
RagasEvaluator — adapts a DataPacket for evaluation with the Ragas library.

This wrapper handles the DataPacket ↔ Ragas dataset conversion.  You supply
pre-configured Ragas metric objects (with their LLM/embedder already set);
this class runs them and maps the results back into EvalScores.

Requires: pip install ragas>=0.2 datasets

Ragas LLM setup (Ollama example):
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm = LangchainLLMWrapper(ChatOllama(model="llama3.2:3b"))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

Ragas LLM setup (OpenAI example):
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

Supported metric name mappings (from ragas result columns → EvalScores fields):
    "faithfulness"      → eval_scores.faithfulness
    "answer_relevancy"  → eval_scores.answer_relevance
    "context_precision" → eval_scores.context_precision

Reference:
  Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  (2023). https://arxiv.org/abs/2309.15217
"""

from __future__ import annotations

import time
from typing import Any

from atomic_rag.schema import DataPacket, EvalScores, TraceEntry

from .base import PipelineEvalBase

# Mapping from ragas result column names to EvalScores field names
_METRIC_MAP: dict[str, str] = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevance",
    "context_precision": "context_precision",
}


class RagasEvaluator(PipelineEvalBase):
    """
    Runs Ragas metrics against a DataPacket and populates eval_scores.

    Parameters
    ----------
    metrics:
        List of pre-configured Ragas Metric objects.  Each metric must have
        its LLM (and optionally embedder) set before being passed here.
    reference:
        Optional ground-truth answer string.  Required for metrics such as
        ``ContextPrecision`` and ``ContextRecall`` that need a reference.

    Example
    -------
    ::

        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from langchain_ollama import ChatOllama

        llm = LangchainLLMWrapper(ChatOllama(model="llama3.2:3b"))
        metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm)]

        evaluator = RagasEvaluator(metrics=metrics)
        result = evaluator.score(packet)

        print(result.eval_scores.faithfulness)
    """

    def __init__(
        self,
        metrics: list[Any],
        reference: str | None = None,
    ) -> None:
        self._metrics = metrics
        self._reference = reference

    def score(self, packet: DataPacket) -> DataPacket:
        try:
            from ragas import evaluate
            from ragas import SingleTurnSample, EvaluationDataset
        except ImportError as exc:
            raise ImportError(
                "Ragas is not installed. Run: pip install 'ragas>=0.2' datasets"
            ) from exc

        t0 = time.monotonic()

        # Build the Ragas sample from the DataPacket
        contexts = [doc.content for doc in packet.documents] if packet.documents else [packet.context]

        sample_kwargs: dict[str, Any] = {
            "user_input": packet.query,
            "retrieved_contexts": contexts,
            "response": packet.answer,
        }
        if self._reference is not None:
            sample_kwargs["reference"] = self._reference

        sample = SingleTurnSample(**sample_kwargs)
        dataset = EvaluationDataset(samples=[sample])

        result = evaluate(dataset=dataset, metrics=self._metrics)
        result_df = result.to_pandas()

        # Map ragas result columns → EvalScores fields
        scores_update: dict[str, float] = {}
        for ragas_col, eval_field in _METRIC_MAP.items():
            if ragas_col in result_df.columns:
                val = result_df[ragas_col].iloc[0]
                if val is not None and not (isinstance(val, float) and val != val):  # not NaN
                    scores_update[eval_field] = round(float(val), 4)

        new_scores = packet.eval_scores.model_copy(update=scores_update)

        duration_ms = (time.monotonic() - t0) * 1000
        entry = TraceEntry(
            phase="evaluation",
            duration_ms=round(duration_ms, 2),
            details={
                "evaluator": "ragas",
                "metrics": [type(m).__name__ for m in self._metrics],
                "scores": scores_update,
            },
        )
        return packet.model_copy(update={"eval_scores": new_scores}).with_trace(entry)
