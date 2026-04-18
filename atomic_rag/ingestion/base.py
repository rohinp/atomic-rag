"""
Abstract base class for all ingestion implementations.

Any ingestor takes a file path and returns a list of Document chunks.
The chunking strategy is an implementation detail — callers only depend on this interface.

To add a new ingestor (e.g. for a different parser):
  1. Subclass IngestorBase
  2. Implement ingest()
  3. The returned Documents must have content, source, and chunk_index set.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from atomic_rag.schema import Document


class IngestorBase(ABC):
    @abstractmethod
    def ingest(self, file_path: str | Path) -> list[Document]:
        """
        Parse a file and return it as an ordered list of Document chunks.

        Args:
            file_path: Path to the file to ingest. The file must exist.

        Returns:
            Non-empty list of Documents ordered by their position in the source.
            Each Document has chunk_index set (0-based), content non-empty,
            and source set to the resolved file path string.

        Raises:
            FileNotFoundError: if file_path does not exist.
            ValueError: if the file cannot be parsed or produces no content.
        """
