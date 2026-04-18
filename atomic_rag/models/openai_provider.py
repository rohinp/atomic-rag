"""
OpenAI model provider — API-based, requires OPENAI_API_KEY.

Install the Python client:
    pip install "atomic-rag[openai]"   (adds the `openai` package)

Set your API key:
    export OPENAI_API_KEY=sk-...
"""

import os

from atomic_rag.models.base import ChatModelBase, EmbedderBase

_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
_DEFAULT_CHAT_MODEL = "gpt-4o-mini"


def _get_client():
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        ) from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )
    return OpenAI(api_key=api_key)


class OpenAIEmbedder(EmbedderBase):
    """
    Embed text using the OpenAI Embeddings API.

    Recommended model: text-embedding-3-small (cost-effective, 1536-dim).
    Alternative: text-embedding-3-large (better quality, higher cost).

    Args:
        model: OpenAI embedding model name.
    """

    def __init__(self, model: str = _DEFAULT_EMBED_MODEL) -> None:
        self.model = model

    def embed(self, text: str) -> list[float]:
        client = _get_client()
        response = client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # OpenAI supports native batching — use it
        client = _get_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


class OpenAIChat(ChatModelBase):
    """
    Generate text using the OpenAI Chat Completions API.

    Recommended model: gpt-4o-mini (fast, cheap, good reasoning).
    Alternative: gpt-4o (best quality, higher cost).

    Args:
        model:       OpenAI chat model name.
        temperature: Sampling temperature. 0.0 for deterministic output.
    """

    def __init__(
        self,
        model: str = _DEFAULT_CHAT_MODEL,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]]) -> str:
        client = _get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
