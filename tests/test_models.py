"""
Tests for atomic_rag.models — EmbedderBase, ChatModelBase, OllamaEmbedder,
OllamaChat, OpenAIEmbedder, OpenAIChat.

All tests run offline. Provider clients (ollama, openai) are mocked so no
real HTTP calls are made. Integration tests that hit real providers live in
tests/integration/.
"""

import builtins
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from atomic_rag.models.base import ChatModelBase, EmbedderBase
from atomic_rag.models.ollama import OllamaChat, OllamaEmbedder
from atomic_rag.models.openai_provider import OpenAIChat, OpenAIEmbedder


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ollama_mock(embedding=None, chat_content=None):
    """Build a mock ollama module with a configurable Client."""
    mock_client = MagicMock()
    if embedding is not None:
        mock_client.embeddings.return_value = {"embedding": embedding}
    if chat_content is not None:
        mock_client.chat.return_value = {"message": {"content": chat_content}}
    mock_ollama = MagicMock()
    mock_ollama.Client.return_value = mock_client
    return mock_ollama, mock_client


def _make_openai_mock(embedding=None, chat_content=None):
    """Build a mock openai module."""
    mock_client = MagicMock()
    if embedding is not None:
        embed_data = MagicMock()
        embed_data.embedding = embedding
        embed_data.index = 0
        mock_client.embeddings.create.return_value = MagicMock(data=[embed_data])
    if chat_content is not None:
        choice = MagicMock()
        choice.message.content = chat_content
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai, mock_client


# ── EmbedderBase ──────────────────────────────────────────────────────────────

class TestEmbedderBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            EmbedderBase()

    def test_embed_batch_default_calls_embed_n_times(self):
        class StubEmbedder(EmbedderBase):
            def embed(self, text):
                return [len(text) * 0.1]

        embedder = StubEmbedder()
        results = embedder.embed_batch(["a", "bb", "ccc"])
        assert results == [[0.1], [0.2], [0.30000000000000004]]


# ── ChatModelBase ─────────────────────────────────────────────────────────────

class TestChatModelBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ChatModelBase()

    def test_complete_wraps_chat(self):
        class StubChat(ChatModelBase):
            def chat(self, messages):
                return messages[0]["content"].upper()

        model = StubChat()
        assert model.complete("hello") == "HELLO"

    def test_complete_sends_user_role(self):
        received = []

        class StubChat(ChatModelBase):
            def chat(self, messages):
                received.extend(messages)
                return "ok"

        StubChat().complete("test prompt")
        assert received[0]["role"] == "user"
        assert received[0]["content"] == "test prompt"


# ── OllamaEmbedder ────────────────────────────────────────────────────────────

class TestOllamaEmbedder:
    def test_embed_returns_vector(self, monkeypatch):
        mock_ollama, mock_client = _make_ollama_mock(embedding=[0.1, 0.2, 0.3])
        monkeypatch.setitem(sys.modules, "ollama", mock_ollama)

        result = OllamaEmbedder().embed("hello world")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_calls_client_with_correct_args(self, monkeypatch):
        mock_ollama, mock_client = _make_ollama_mock(embedding=[0.0])
        monkeypatch.setitem(sys.modules, "ollama", mock_ollama)

        OllamaEmbedder(model="nomic-embed-text").embed("test text")
        mock_client.embeddings.assert_called_once_with(
            model="nomic-embed-text", prompt="test text"
        )

    def test_embed_batch_calls_embed_for_each_text(self, monkeypatch):
        mock_ollama, mock_client = _make_ollama_mock(embedding=[0.5])
        monkeypatch.setitem(sys.modules, "ollama", mock_ollama)

        OllamaEmbedder().embed_batch(["a", "b", "c"])
        assert mock_client.embeddings.call_count == 3

    def test_custom_host_passed_to_client(self, monkeypatch):
        mock_ollama, _ = _make_ollama_mock(embedding=[0.0])
        monkeypatch.setitem(sys.modules, "ollama", mock_ollama)

        OllamaEmbedder(host="http://remote:11434").embed("x")
        mock_ollama.Client.assert_called_with(host="http://remote:11434")

    def test_import_error_when_ollama_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ollama", None)
        with pytest.raises(ImportError, match="pip install ollama"):
            OllamaEmbedder().embed("x")

    def test_default_model_is_nomic_embed_text(self):
        assert OllamaEmbedder().model == "nomic-embed-text"

    def test_default_host_is_localhost(self):
        assert "localhost" in OllamaEmbedder().host


# ── OllamaChat ────────────────────────────────────────────────────────────────

class TestOllamaChat:
    def test_chat_returns_assistant_content(self, monkeypatch):
        mock_ollama, _ = _make_ollama_mock(chat_content="Paris")
        monkeypatch.setitem(sys.modules, "ollama", mock_ollama)

        result = OllamaChat().chat([{"role": "user", "content": "Capital of France?"}])
        assert result == "Paris"

    def test_chat_passes_messages(self, monkeypatch):
        mock_ollama, mock_client = _make_ollama_mock(chat_content="ok")
        monkeypatch.setitem(sys.modules, "ollama", mock_ollama)

        messages = [{"role": "user", "content": "hello"}]
        OllamaChat(model="llama3.2:3b").chat(messages)
        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs["messages"] == messages
        assert call_kwargs.kwargs["model"] == "llama3.2:3b"

    def test_temperature_passed_in_options(self, monkeypatch):
        mock_ollama, mock_client = _make_ollama_mock(chat_content="ok")
        monkeypatch.setitem(sys.modules, "ollama", mock_ollama)

        OllamaChat(temperature=0.7).chat([{"role": "user", "content": "x"}])
        options = mock_client.chat.call_args.kwargs["options"]
        assert options["temperature"] == 0.7

    def test_complete_convenience_method(self, monkeypatch):
        mock_ollama, _ = _make_ollama_mock(chat_content="answer")
        monkeypatch.setitem(sys.modules, "ollama", mock_ollama)

        result = OllamaChat().complete("question")
        assert result == "answer"

    def test_import_error_when_ollama_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ollama", None)
        with pytest.raises(ImportError, match="pip install ollama"):
            OllamaChat().chat([])

    def test_default_model_is_llama(self):
        assert "llama" in OllamaChat().model


# ── OpenAIEmbedder ────────────────────────────────────────────────────────────

class TestOpenAIEmbedder:
    def test_embed_returns_vector(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_openai, _ = _make_openai_mock(embedding=[0.1, 0.2])
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        result = OpenAIEmbedder().embed("hello")
        assert result == [0.1, 0.2]

    def test_embed_calls_api_with_correct_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_openai, mock_client = _make_openai_mock(embedding=[0.0])
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        OpenAIEmbedder(model="text-embedding-3-small").embed("text")
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="text"
        )

    def test_embed_batch_uses_native_batching(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_client = MagicMock()
        items = [MagicMock(embedding=[float(i)], index=i) for i in range(3)]
        mock_client.embeddings.create.return_value = MagicMock(data=items)
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        OpenAIEmbedder().embed_batch(["a", "b", "c"])
        # Native batching: called once, not three times
        assert mock_client.embeddings.create.call_count == 1

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_openai, _ = _make_openai_mock(embedding=[0.0])
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            OpenAIEmbedder().embed("text")

    def test_import_error_when_openai_not_installed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="pip install openai"):
            OpenAIEmbedder().embed("x")


# ── OpenAIChat ────────────────────────────────────────────────────────────────

class TestOpenAIChat:
    def test_chat_returns_content(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_openai, _ = _make_openai_mock(chat_content="Berlin")
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        result = OpenAIChat().chat([{"role": "user", "content": "Capital of Germany?"}])
        assert result == "Berlin"

    def test_chat_passes_temperature(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_openai, mock_client = _make_openai_mock(chat_content="ok")
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        OpenAIChat(temperature=0.5).chat([{"role": "user", "content": "x"}])
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_openai, _ = _make_openai_mock(chat_content="ok")
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            OpenAIChat().chat([])

    def test_import_error_when_openai_not_installed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="pip install openai"):
            OpenAIChat().chat([])
