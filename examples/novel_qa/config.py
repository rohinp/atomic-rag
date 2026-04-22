"""
Model provider configuration for the novel Q&A example.

Default: Ollama (local, no API key required).

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull the required models:
           ollama pull nomic-embed-text
           ollama pull llama3.2:3b

To use OpenAI instead:
    1. pip install openai
    2. export OPENAI_API_KEY=sk-...
    3. Comment out the Ollama block below and uncomment the OpenAI block.
"""

# ── Ollama (default — local, no API key) ──────────────────────────────────────
from atomic_rag.models.ollama import OllamaChat, OllamaEmbedder

EMBEDDER = OllamaEmbedder(model="nomic-embed-text")
CHAT_MODEL = OllamaChat(model="llama3.2:3b", temperature=0.0)

# ── OpenAI (uncomment to use instead) ─────────────────────────────────────────
# from atomic_rag.models.openai_provider import OpenAIChat, OpenAIEmbedder
#
# EMBEDDER = OpenAIEmbedder(model="text-embedding-3-small")
# CHAT_MODEL = OpenAIChat(model="gpt-4o-mini", temperature=0.0)
