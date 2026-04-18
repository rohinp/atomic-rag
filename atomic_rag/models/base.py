"""
Abstract interfaces for embedding models and chat/completion models.

Every model provider (Ollama, OpenAI, etc.) implements these two interfaces.
The rest of the pipeline only depends on these — never on a specific provider.
"""

from abc import ABC, abstractmethod


class EmbedderBase(ABC):
    """Converts text into a dense vector (embedding)."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a float vector."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings. Default: sequential calls to embed().
        Providers that support native batching should override this.
        """
        return [self.embed(t) for t in texts]


class ChatModelBase(ABC):
    """Generates text from a list of chat messages."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Send a list of messages and return the assistant's reply.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."} dicts.

        Returns:
            The assistant's response as a plain string.
        """

    def complete(self, prompt: str) -> str:
        """Convenience wrapper: single user turn with no history."""
        return self.chat([{"role": "user", "content": prompt}])
