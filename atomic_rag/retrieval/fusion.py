"""
Reciprocal Rank Fusion (RRF) — pure function, no dependencies.

RRF merges multiple ranked lists into a single ranking without requiring
score normalisation. Because it operates on ranks (not raw scores), it
handles the incompatible scales of BM25 and cosine similarity naturally.

Reference: Cormack, Clarke & Buettcher (2009).
  "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
  ACM SIGIR. https://dl.acm.org/doi/10.1145/1571941.1572114
"""

from atomic_rag.schema import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Combine multiple ranked lists into one via Reciprocal Rank Fusion.

    Each document's RRF score is the sum of 1/(k + rank) across all lists
    it appears in (1-based rank). Documents that appear in more lists, or
    rank higher in any list, get a higher fused score.

    The k=60 default comes from the original paper — it smooths the curve
    so that rank 1 is not disproportionately favoured over ranks 2–10.

    Args:
        ranked_lists: Each list is sorted best-first. A document may appear
                      in multiple lists (identified by Document.id).
        k:            Smoothing constant. Higher k = gentler rank weighting.

    Returns:
        Deduplicated documents sorted by fused RRF score descending.
        Document.score is set to the RRF score (not the original scores).
        Returns [] if all input lists are empty.
    """
    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc.id] = doc  # keep most-recent copy (scores may differ)

    return [
        doc.model_copy(update={"score": rrf_scores[doc.id]})
        for doc in sorted(doc_map.values(), key=lambda d: rrf_scores[d.id], reverse=True)
    ]
