"""
Sentence-level context compressor.

For each retrieved Document, splits the content into segments and drops any
segment whose cosine similarity to the query falls below `threshold`. This
removes padding sentences that happened to land in the same chunk as a
relevant sentence — they consume tokens without helping the LLM.

Segment strategy by content type:
  - Prose (no metadata["language"])  → split on sentence boundaries
  - Code (metadata["language"] set)  → keep the whole chunk as one unit.
    A partial function body is syntactically invalid and semantically useless.

The final context string is assembled as:

    [Source: path/to/file.py  L42–61]
    <compressed content>

    ---

    [Source: report.pdf  chunk 3]
    <compressed content>

Source headers let the LLM cite accurately without hallucinating file paths.

Performance note: sentences within each document are embedded in a single
batch call to minimise round-trips to the embedding server. The query is
embedded once and reused across all documents.
"""

import math
import re
import time

from atomic_rag.context.base import CompressorBase
from atomic_rag.models.base import EmbedderBase
from atomic_rag.schema import DataPacket, Document, TraceEntry

# Splits on . ! ? followed by whitespace + uppercase — avoids splitting
# on decimal numbers and common abbreviations like "e.g." that don't
# precede an uppercase letter.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """
    Split prose text into sentences using punctuation heuristics.

    Not perfect — no NLP library dependency by design. Handles the common
    cases well enough for context compression purposes.
    """
    parts = _SENT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two equal-length vectors. Returns 0.0 for
    zero vectors rather than raising ZeroDivisionError.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _segments(doc: Document) -> list[str]:
    """
    Return the segments to score for a document.

    Code chunks are returned as a single segment — splitting a function
    body into sentences and potentially discarding half of it would produce
    syntactically broken, useless context.
    """
    if doc.metadata.get("language"):
        return [doc.content]
    return split_sentences(doc.content)


def _source_label(doc: Document) -> str:
    """Human-readable source header for a document chunk."""
    parts = [f"Source: {doc.source}"]
    if "start_line" in doc.metadata and "end_line" in doc.metadata:
        parts.append(f"L{doc.metadata['start_line']}–{doc.metadata['end_line']}")
    elif doc.chunk_index is not None:
        parts.append(f"chunk {doc.chunk_index}")
    return "  ".join(parts)


class SentenceCompressor(CompressorBase):
    """
    Drop low-relevance sentences from retrieved documents.

    Each sentence is embedded and its cosine similarity to the query is
    computed. Sentences below `threshold` are discarded. At least
    `min_sentences` sentences are always kept per document (even if all
    fall below threshold) so the context is never completely empty.

    Args:
        embedder:      The same embedder used for retrieval — keeps the
                       vector space consistent.
        threshold:     Cosine similarity cutoff in [0, 1]. Higher = stricter
                       filtering. 0.5 is a reasonable starting point; tune
                       down if relevant context is being dropped.
        min_sentences: Minimum segments to keep per document regardless of
                       score. Prevents a document from disappearing entirely.
    """

    def __init__(
        self,
        embedder: EmbedderBase,
        threshold: float = 0.5,
        min_sentences: int = 1,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if min_sentences < 1:
            raise ValueError(f"min_sentences must be >= 1, got {min_sentences}")
        self.embedder = embedder
        self.threshold = threshold
        self.min_sentences = min_sentences

    def compress(self, packet: DataPacket) -> DataPacket:
        """
        Filter each retrieved document to its most query-relevant sentences.

        Returns a new DataPacket with context set to the compressed string.
        Appends a TraceEntry with before/after sentence counts and the
        percentage of content removed.
        """
        t0 = time.monotonic()

        if not packet.documents:
            entry = TraceEntry(
                phase="context",
                duration_ms=0.0,
                details={
                    "documents": 0,
                    "sentences_before": 0,
                    "sentences_after": 0,
                    "reduction_pct": 0.0,
                    "threshold": self.threshold,
                },
            )
            return packet.with_trace(entry)

        query_embedding = self.embedder.embed(packet.query)

        compressed_blocks: list[str] = []
        total_before = 0
        total_after = 0

        for doc in packet.documents:
            segments = _segments(doc)
            total_before += len(segments)

            if not segments:
                continue

            kept = self._filter_segments(segments, query_embedding)
            total_after += len(kept)

            if kept:
                label = _source_label(doc)
                compressed_blocks.append(f"[{label}]\n" + " ".join(kept))

        context = "\n\n---\n\n".join(compressed_blocks)
        reduction = round((1 - total_after / max(total_before, 1)) * 100, 1)

        duration_ms = (time.monotonic() - t0) * 1000
        entry = TraceEntry(
            phase="context",
            duration_ms=round(duration_ms, 2),
            details={
                "documents": len(packet.documents),
                "sentences_before": total_before,
                "sentences_after": total_after,
                "reduction_pct": reduction,
                "threshold": self.threshold,
            },
        )

        return packet.model_copy(update={"context": context}).with_trace(entry)

    def _filter_segments(
        self, segments: list[str], query_embedding: list[float]
    ) -> list[str]:
        """
        Keep segments whose cosine similarity to the query meets the threshold.
        Always keeps at least min_sentences segments.
        """
        embeddings = self.embedder.embed_batch(segments)
        scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]

        kept = [s for s, score in zip(segments, scores) if score >= self.threshold]

        if len(kept) < self.min_sentences:
            # Fall back: keep the highest-scoring segments up to min_sentences
            ranked = sorted(zip(scores, segments), reverse=True)
            kept = [s for _, s in ranked[: self.min_sentences]]

        return kept
