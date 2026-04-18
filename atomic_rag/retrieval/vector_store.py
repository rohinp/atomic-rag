"""
ChromaDB-backed vector store.

Stores document embeddings and runs approximate nearest-neighbour (ANN) search.
Returns Documents with score = cosine similarity (1.0 = identical, 0.0 = orthogonal).

The client is injected via _client for testing — production code leaves it None
and the store creates its own ChromaDB client lazily on first use.
"""

from typing import Any

from atomic_rag.schema import Document


class ChromaVectorStore:
    """
    Wrapper around a ChromaDB collection.

    Args:
        collection_name: Name of the ChromaDB collection. One collection per index.
        persist_path:    Directory for persistent storage. None = in-memory.
        _client:         Injected ChromaDB client (for testing). Leave as None in production.
    """

    def __init__(
        self,
        collection_name: str = "atomic_rag",
        persist_path: str | None = None,
        _client: Any = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_path = persist_path
        self._client = _client
        self._collection: Any = None

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, docs: list[Document], embeddings: list[list[float]]) -> None:
        """
        Store documents with their pre-computed embeddings.

        Args:
            docs:       Documents to store. ids must be unique.
            embeddings: One embedding per document, same order.
        """
        if not docs:
            return
        collection = self._get_collection()
        collection.add(
            ids=[d.id for d in docs],
            embeddings=embeddings,
            documents=[d.content for d in docs],
            metadatas=[self._to_metadata(d) for d in docs],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[Document]:
        """
        Find the top_k documents most similar to query_embedding.

        Returns Documents with score = cosine similarity [0, 1].
        Returns fewer than top_k if the collection has fewer documents.
        """
        collection = self._get_collection()
        n = min(top_k, self._count())
        if n == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for doc_id, content, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns L2 or cosine distance depending on collection config.
            # We configure cosine space, so: similarity = 1 - distance
            score = max(0.0, 1.0 - float(distance))
            source = metadata.pop("_source", "")
            chunk_index = metadata.pop("_chunk_index", None)
            docs.append(Document(
                id=doc_id,
                content=content,
                source=source,
                chunk_index=int(chunk_index) if chunk_index is not None else None,
                metadata=metadata,
                score=score,
            ))
        return docs

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        client = self._client or self._make_client()
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _make_client(self) -> Any:
        try:
            import chromadb  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install with: pip install chromadb"
            ) from e
        if self._persist_path:
            return chromadb.PersistentClient(path=self._persist_path)
        try:
            return chromadb.EphemeralClient()
        except AttributeError:
            return chromadb.Client()  # older chromadb versions

    def _count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    @staticmethod
    def _to_metadata(doc: Document) -> dict:
        # ChromaDB metadata values must be str | int | float | bool.
        # Prefix reserved fields with _ to avoid collisions with doc.metadata.
        meta: dict = {"_source": doc.source}
        if doc.chunk_index is not None:
            meta["_chunk_index"] = doc.chunk_index
        for k, v in doc.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        return meta
