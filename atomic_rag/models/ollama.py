"""
Ollama model provider — local, no API key required.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull the models you need:
           ollama pull nomic-embed-text   # embeddings
           ollama pull llama3.2:3b        # fast chat (testing)
           ollama pull llama3.1:8b        # better quality chat

Install the Python client:
    pip install "atomic-rag[ollama]"   (adds the `ollama` package)
"""

from atomic_rag.models.base import ChatModelBase, EmbedderBase

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"
_DEFAULT_CHAT_MODEL = "llama3.2:3b"


def _get_client(host: str):
    try:
        import ollama  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "ollama package is required. Install with: pip install ollama"
        ) from e
    return ollama.Client(host=host)


class OllamaEmbedder(EmbedderBase):
    """
    Embed text using a locally-running Ollama model.

    Recommended model: nomic-embed-text (768-dim, fast, strong on code and prose).
    Alternative: mxbai-embed-large (1024-dim, higher quality, slower).

    Args:
        model: Ollama model name. Must be pulled before use.
        host:  Ollama server URL. Default assumes local install.
    """

    def __init__(
        self,
        model: str = _DEFAULT_EMBED_MODEL,
        host: str = _DEFAULT_HOST,
    ) -> None:
        self.model = model
        self.host = host

    def embed(self, text: str) -> list[float]:
        client = _get_client(self.host)
        response = client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama doesn't support native batching — sequential is correct
        client = _get_client(self.host)
        return [
            client.embeddings(model=self.model, prompt=t)["embedding"]
            for t in texts
        ]


class OllamaChat(ChatModelBase):
    """
    Generate text using a locally-running Ollama chat model.

    Recommended models:
      llama3.2:3b  — fast, good enough for testing and development
      llama3.1:8b  — better reasoning, slower, recommended for production

    Args:
        model:       Ollama model name. Must be pulled before use.
        host:        Ollama server URL.
        temperature: Sampling temperature. 0.0 for deterministic output.
    """

    def __init__(
        self,
        model: str = _DEFAULT_CHAT_MODEL,
        host: str = _DEFAULT_HOST,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]]) -> str:
        client = _get_client(self.host)
        response = client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]
