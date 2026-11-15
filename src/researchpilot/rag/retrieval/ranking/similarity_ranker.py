
"""Similarity-based retrieval ranking implementation."""

from researchpilot.rag.retrieval.ranking.base import RankingStrategy
from researchpilot.rag.vectorstore.models import VectorSearchResult


class SimilarityRanker(RankingStrategy):
    """Rank retrieval results by similarity score."""

    def rank(
        self,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        """Return results ordered by descending similarity score."""
        return sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )
