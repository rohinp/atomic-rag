"""
LLMGenerator — produces a grounded answer from query + context.

The context arriving here is the Phase 4 output: already filtered to
high-relevance sentences with source labels.  The prompt instructs the LLM
to stay within the context and cite sources so answers are auditable.
"""

from __future__ import annotations

from atomic_rag.models.base import ChatModelBase

from .base import GeneratorBase

_DEFAULT_TEMPLATE = (
    "You are a helpful assistant that answers questions based strictly on the "
    "provided context.  Do not add information that is not in the context.  "
    "If the context contains source labels like '[Source: ...]', you may cite "
    "them in your answer.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


class LLMGenerator(GeneratorBase):
    """
    Generates a final answer grounded in the retrieved context.

    Parameters
    ----------
    chat_model:
        Any ChatModelBase (OllamaChat, OpenAIChat, …).
    prompt_template:
        Override the default prompt.  Must contain ``{query}`` and ``{context}``.
    """

    def __init__(
        self,
        chat_model: ChatModelBase,
        prompt_template: str | None = None,
    ) -> None:
        self._chat = chat_model
        self._template = prompt_template or _DEFAULT_TEMPLATE

    def generate(self, query: str, context: str) -> str:
        prompt = self._template.format(query=query, context=context)
        return self._chat.complete(prompt).strip()
