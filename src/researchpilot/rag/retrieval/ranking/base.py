
"""Base interface for retrieval ranking strategies."""

from abc import ABC, abstractmethod

from researchpilot.rag.vectorstore.models import VectorSearchResult


class RankingStrategy(ABC):
    """Define the interface for retrieval ranking strategies."""

    @abstractmethod
    def rank(
        self,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        """Rank retrieved candidates and return ordered results."""
        pass
