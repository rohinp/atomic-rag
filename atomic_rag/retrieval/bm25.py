"""
BM25 keyword retriever.

Complements vector search by matching exact terms — function names, acronyms,
version numbers, and error codes that embeddings tend to blur together.

Uses rank_bm25's BM25Okapi implementation, which applies IDF weighting and
document-length normalisation. The rank_bm25 import is lazy so the module can
be imported without the package installed (fails only when add() is called).
"""

from typing import Any

from atomic_rag.schema import Document


class BM25Retriever:
    """
    BM25 keyword retriever backed by rank_bm25.

    Usage:
        retriever = BM25Retriever()
        retriever.add(docs)               # build index
        results = retriever.search("query text", top_k=50)
    """

    def __init__(self) -> None:
        self._bm25: Any = None
        self._docs: list[Document] = []

    def add(self, docs: list[Document]) -> None:
        """
        Build a BM25 index over docs.

        Tokenises by whitespace after lowercasing. This is intentionally
        simple — BM25's strength is exact token matching, not semantic
        understanding, so aggressive tokenisation would be counter-productive.

        Args:
            docs: Documents to index. Replaces any existing index.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "rank-bm25 is required for BM25Retriever. "
                "Install with: pip install rank-bm25"
            ) from e
        self._docs = list(docs)
        tokenised = [d.content.lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int) -> list[Document]:
        """
        Return the top_k documents by BM25 score.

        Documents with a score of 0.0 are excluded — they contain none of
        the query terms and would only add noise to the fused ranking.

        Args:
            query:  Raw query string. Tokenised the same way as documents.
            top_k:  Maximum number of results.

        Returns:
            Documents sorted descending by BM25 score, score set on each.
            Returns [] if the index is empty or no terms match.
        """
        if self._bm25 is None or not self._docs:
            return []

        tokens = query.lower().split()
        raw_scores: list[float] = self._bm25.get_scores(tokens).tolist()

        ranked = sorted(
            ((score, doc) for score, doc in zip(raw_scores, self._docs) if score > 0.0),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            doc.model_copy(update={"score": score})
            for score, doc in ranked[:top_k]
        ]
