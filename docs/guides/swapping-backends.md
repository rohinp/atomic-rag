# Swapping Backends

Every backend in atomic-rag is an injected dependency behind a small abstract interface. Swapping one out — replacing ChromaDB with Qdrant, Ollama with OpenAI, MarkItDown with Docling — means implementing the right interface and passing it in. Nothing else changes.

---

## Vector store: ChromaDB → Qdrant

**When to switch**: Qdrant scales better for large corpora (millions of vectors), supports distributed deployments, and has richer filtering. ChromaDB is simpler for local development.

### The interface

`ChromaVectorStore` is the only vector-store implementation today. `HybridRetriever` accepts it as an injected `vector_store` argument. To use Qdrant, implement the same two-method surface:

```python
# The methods HybridRetriever calls:
#   store.add(docs: list[Document], embeddings: list[list[float]]) -> None
#   store.search(query_embedding: list[float], top_k: int) -> list[Document]
```

### Qdrant implementation

```bash
pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from atomic_rag.schema import Document
import uuid


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = "atomic_rag",
        url: str | None = None,          # None = in-memory
        api_key: str | None = None,
        vector_size: int = 768,          # match your embedder's output dim
    ):
        self._name = collection_name
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(":memory:")

        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def add(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.id)),
                vector=emb,
                payload={
                    "_id": doc.id,
                    "_content": doc.content,
                    "_source": doc.source,
                    "_chunk_index": doc.chunk_index,
                    **{k: v for k, v in doc.metadata.items()
                       if isinstance(v, (str, int, float, bool))},
                },
            )
            for doc, emb in zip(docs, embeddings)
        ]
        self._client.upsert(collection_name=self._name, points=points)

    def search(self, query_embedding: list[float], top_k: int) -> list[Document]:
        hits = self._client.search(
            collection_name=self._name,
            query_vector=query_embedding,
            limit=top_k,
        )
        docs = []
        for hit in hits:
            p = hit.payload
            docs.append(Document(
                id=p["_id"],
                content=p["_content"],
                source=p["_source"],
                chunk_index=p.get("_chunk_index"),
                metadata={k: v for k, v in p.items() if not k.startswith("_")},
                score=hit.score,
            ))
        return docs
```

### Plug it in

```python
from atomic_rag.retrieval import HybridRetriever
from atomic_rag.models.ollama import OllamaEmbedder

retriever = HybridRetriever(
    embedder=OllamaEmbedder(),
    vector_store=QdrantVectorStore(
        url="http://localhost:6333",  # or None for in-memory
        vector_size=768,              # nomic-embed-text outputs 768 dims
    ),
)
```

Everything downstream (BM25, RRF, reranker) is unchanged.

---

## Document parser: MarkItDown → Docling

**When to switch**: Docling (IBM Research) has stronger table extraction and layout analysis for complex PDFs. MarkItDown is simpler and handles more file types (PPTX, XLSX, etc.).

### The interface

Any parser that produces Markdown text can be dropped into `MarkItDownIngestor` via the `_converter` parameter — or you can write a new `IngestorBase` subclass that skips Markdown entirely and chunks directly.

### Option A — drop-in converter replacement

`MarkItDownIngestor` calls `converter.convert(path).text_content`. Wrap Docling to match:

```bash
pip install docling
```

```python
from pathlib import Path
from docling.document_converter import DocumentConverter as DoclingConverter


class DoclingAdapter:
    """Wraps Docling to match MarkItDown's convert() interface."""

    def __init__(self):
        self._conv = DoclingConverter()

    def convert(self, path: str):
        result = self._conv.convert(path)

        class _Result:
            text_content = result.document.export_to_markdown()

        return _Result()
```

```python
from atomic_rag.ingestion import MarkItDownIngestor

ingestor = MarkItDownIngestor(_converter=DoclingAdapter())
docs = ingestor.ingest("reports/q4-2024.pdf")
```

### Option B — full IngestorBase subclass

For more control over chunking strategy:

```python
from pathlib import Path
from atomic_rag.ingestion.base import IngestorBase
from atomic_rag.ingestion.chunker import MarkdownChunker
from atomic_rag.schema import Document


class DoclingIngestor(IngestorBase):
    def __init__(self, chunker=None):
        self.chunker = chunker or MarkdownChunker()

    def ingest(self, file_path: str | Path) -> list[Document]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise ImportError("pip install docling") from e

        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        result = DocumentConverter().convert(str(path))
        markdown = result.document.export_to_markdown()

        if not markdown.strip():
            raise ValueError(f"Docling produced no content for: {path}")

        return self.chunker.chunk(markdown, source=str(path))
```

---

## Embedding model: Ollama → OpenAI

Both implement `EmbedderBase` — swap the import, nothing else changes.

```python
# Before (Ollama, local)
from atomic_rag.models.ollama import OllamaEmbedder
embedder = OllamaEmbedder(model="nomic-embed-text")

# After (OpenAI API)
from atomic_rag.models.openai_provider import OpenAIEmbedder
embedder = OpenAIEmbedder(model="text-embedding-3-small")
# Reads OPENAI_API_KEY from environment
```

Pass the new embedder wherever `EmbedderBase` is accepted:

```python
from atomic_rag.retrieval import HybridRetriever
from atomic_rag.context import SentenceCompressor

retriever = HybridRetriever(embedder=embedder)
compressor = SentenceCompressor(embedder=embedder, threshold=0.5)
```

> Use the **same embedder instance** for both retrieval and compression. Mixing providers creates an inconsistent vector space — cosine similarity between an Ollama query vector and an OpenAI document vector is meaningless.

### Embedding model dimension notes

| Model | Dimensions |
|---|---|
| `nomic-embed-text` (Ollama) | 768 |
| `text-embedding-3-small` (OpenAI) | 1536 |
| `text-embedding-3-large` (OpenAI) | 3072 |

If you change embedder, recreate the vector store — dimensions must match.

---

## Chat model: Ollama → OpenAI

Both implement `ChatModelBase`:

```python
# Before (Ollama, local)
from atomic_rag.models.ollama import OllamaChat
chat = OllamaChat(model="llama3.2:3b", temperature=0.0)

# After (OpenAI API)
from atomic_rag.models.openai_provider import OpenAIChat
chat = OpenAIChat(model="gpt-4o-mini", temperature=0.0)
```

Pass the new model to any component that accepts `ChatModelBase`:

```python
from atomic_rag.query import HyDEExpander, MultiQueryExpander
from atomic_rag.agent import AgentRunner, LLMEvaluator, LLMGenerator
from atomic_rag.evaluation import LLMFaithfulnessScorer

expander = HyDEExpander(chat_model=chat)
runner = AgentRunner(
    evaluator=LLMEvaluator(chat_model=chat),
    generator=LLMGenerator(chat_model=chat),
)
scorer = LLMFaithfulnessScorer(chat_model=chat)
```

---

## Adding a new model provider

Implement `EmbedderBase` and/or `ChatModelBase` and inject it. Example for a hypothetical `AcmeAPI`:

```python
from atomic_rag.models.base import EmbedderBase, ChatModelBase


class AcmeEmbedder(EmbedderBase):
    def __init__(self, api_key: str):
        self._key = api_key

    def embed(self, text: str) -> list[float]:
        import acme_sdk
        return acme_sdk.embed(text, api_key=self._key)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import acme_sdk
        return acme_sdk.embed_batch(texts, api_key=self._key)


class AcmeChat(ChatModelBase):
    def __init__(self, api_key: str, model: str = "acme-7b"):
        self._key = api_key
        self._model = model

    def chat(self, messages: list[dict]) -> str:
        import acme_sdk
        return acme_sdk.chat(messages, model=self._model, api_key=self._key)
```

No changes to any pipeline component.

---

## Cross-encoder reranker: ms-marco-MiniLM → BGE-Reranker

`CrossEncoderReranker` accepts a model name string. Pass any `sentence-transformers`-compatible cross-encoder:

```python
from atomic_rag.retrieval.reranker import CrossEncoderReranker

# Default
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

# Stronger (slower)
reranker = CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")

# Multilingual
reranker = CrossEncoderReranker(model="cross-encoder/msmarco-MiniLM-L6-en-de-v1")

retriever = HybridRetriever(embedder=embedder, reranker=reranker)
```

---

## Quick reference

| What to swap | Interface to implement | Injected into |
|---|---|---|
| Vector store | `add(docs, embeddings)` + `search(embedding, k)` | `HybridRetriever(vector_store=...)` |
| Document parser | `IngestorBase.ingest()` or adapter with `.convert().text_content` | `MarkItDownIngestor(_converter=...)` |
| Embedding model | `EmbedderBase` | `HybridRetriever`, `SentenceCompressor`, `EmbeddingAnswerRelevance` |
| Chat model | `ChatModelBase` | `HyDEExpander`, `MultiQueryExpander`, `LLMEvaluator`, `LLMGenerator`, `LLMFaithfulnessScorer` |
| Cross-encoder | model name string (sentence-transformers) | `CrossEncoderReranker(model=...)` |
