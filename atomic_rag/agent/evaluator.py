"""
LLMEvaluator — grades retrieved context quality using a language model.

Part of the Corrective RAG (C-RAG) loop.  Before generating an answer, the
evaluator asks the LLM whether the context is sufficient.  If the score is
below AgentRunner.threshold, a fallback is triggered instead of generating a
potentially hallucinated answer.

Prompt design follows the graded relevance approach from:
  Yan et al., "Corrective Retrieval Augmented Generation" (2024).
  https://arxiv.org/abs/2401.15884
"""

from __future__ import annotations

import re

from atomic_rag.models.base import ChatModelBase

from .base import EvaluatorBase

_DEFAULT_TEMPLATE = (
    "You are evaluating whether a retrieved context is sufficient to answer a question.\n\n"
    "Question: {query}\n\n"
    "Context:\n{context}\n\n"
    "Rate how well the context answers the question on a scale of 0.0 to 1.0:\n"
    "- 1.0: the context directly and completely answers the question\n"
    "- 0.5: the context is partially relevant but incomplete\n"
    "- 0.0: the context is irrelevant or does not contain the answer\n\n"
    "Respond with ONLY a decimal number between 0.0 and 1.0. No explanation."
)

_FLOAT_RE = re.compile(r"\b(1\.0|0\.\d+|\d\.\d+)\b")


def _parse_score(text: str) -> float | None:
    """Extract the first valid [0, 1] float from LLM output."""
    text = text.strip()
    # direct float parse first
    try:
        val = float(text)
        return max(0.0, min(1.0, val))
    except ValueError:
        pass
    # regex fallback for embedded numbers
    match = _FLOAT_RE.search(text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    # YES/NO fallback for models that ignore the format
    upper = text.upper()
    if upper.startswith("YES"):
        return 1.0
    if upper.startswith("NO"):
        return 0.0
    return None


class LLMEvaluator(EvaluatorBase):
    """
    Uses an LLM to score whether the context answers the query.

    Parameters
    ----------
    chat_model:
        Any ChatModelBase (OllamaChat, OpenAIChat, …).
    prompt_template:
        Override the default prompt.  Must contain ``{query}`` and ``{context}``.
    default_score:
        Score to return when the LLM output cannot be parsed (default 0.5).
    """

    def __init__(
        self,
        chat_model: ChatModelBase,
        prompt_template: str | None = None,
        default_score: float = 0.5,
    ) -> None:
        self._chat = chat_model
        self._template = prompt_template or _DEFAULT_TEMPLATE
        self._default = default_score

    def evaluate(self, query: str, context: str) -> float:
        if not context.strip():
            return 0.0

        prompt = self._template.format(query=query, context=context)
        raw = self._chat.complete(prompt)
        score = _parse_score(raw)
        return score if score is not None else self._default
