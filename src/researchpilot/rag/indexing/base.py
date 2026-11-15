
"""Abstract interface for evidence indexing."""

from abc import ABC, abstractmethod

from researchpilot.rag.indexing.models import (
    IndexingRequest,
    IndexingResult,
)
from researchpilot.retrieval.fulltext.models import Evidence


class EvidenceIndexer(ABC):
    """Define the contract for evidence indexing."""

    @abstractmethod
    def index_evidence(
        self,
        evidence: list[Evidence],
        request: IndexingRequest | None = None,
    ) -> IndexingResult:
        """Embed and index evidence records."""
