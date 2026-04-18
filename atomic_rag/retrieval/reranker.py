"""
Cross-encoder reranker.

A cross-encoder reads the query and document *together* (not separately like
bi-encoders used for vector search). This gives much higher accuracy but is too
slow to run over thousands of documents — so it's applied only to the small set
of candidates that survived the first-stage hybrid retrieval.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fast (MiniLM architecture), 6 layers
  - Trained on MS MARCO passage ranking
  - Good balance of speed and quality for general-purpose reranking

Alternative: BAAI/bge-reranker-base — stronger on multi-lingual and
             technical corpora (code, scientific text)
"""

from typing import Any

from atomic_rag.schema import Document


class CrossEncoderReranker:
    """
    Rerank documents using a cross-encoder model from sentence-transformers.

    Args:
        model:          HuggingFace model name or local path.
        _model_instance: Injected model instance for testing. Leave None in production.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        _model_instance: Any = None,
    ) -> None:
        self._model_name = model
        self._model = _model_instance

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """
        Score each (query, document) pair and return the top_k highest-scoring docs.

        The score written to Document.score is the raw cross-encoder logit.
        This is NOT a probability — it is only meaningful for ranking, not as
        an absolute relevance threshold.

        Args:
            query:     The user query.
            documents: Candidate documents from first-stage retrieval.
            top_k:     Number of documents to return.

        Returns:
            Documents sorted descending by cross-encoder score, score updated.
        """
        if not documents:
            return []

        model = self._get_model()
        pairs = [(query, doc.content) for doc in documents]
        scores: list[float] = model.predict(pairs).tolist()

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [
            doc.model_copy(update={"score": float(score)})
            for doc, score in ranked[:top_k]
        ]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model = CrossEncoder(self._model_name)
        return self._model
