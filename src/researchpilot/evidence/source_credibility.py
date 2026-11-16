"""Academic source credibility scoring utilities."""

from datetime import datetime

from researchpilot.evidence.source_models import (
    SourceCredibilityScore,
)
from researchpilot.retrieval.models import Paper


class SourceCredibilityEvaluator:
    """Evaluate academic source credibility using metadata heuristics."""

    def evaluate(self, paper: Paper) -> SourceCredibilityScore:
        """Calculate source credibility from available paper metadata."""

        doi_score = 1.0 if paper.doi else 0.0

        citation_score = self._calculate_citation_score(
            paper.citation_count
        )

        recency_score = self._calculate_recency_score(
            paper.publication_year
        )

        metadata_score = self._calculate_metadata_score(paper)

        overall_score = (
            doi_score
            + citation_score
            + recency_score
            + metadata_score
        ) / 4

        return SourceCredibilityScore(
            doi_score=doi_score,
            citation_score=citation_score,
            recency_score=recency_score,
            metadata_score=metadata_score,
            overall_score=overall_score,
        )

    @staticmethod
    def _calculate_citation_score(
        citation_count: int | None,
    ) -> float:
        """Normalize citation count using a logarithmic scale."""

        if citation_count is None or citation_count <= 0:
            return 0.0

        return min(
            1.0,
            citation_count.bit_length() / 10,
        )

    @staticmethod
    def _calculate_recency_score(
        publication_year: int | None,
    ) -> float:
        """Score publication recency over a ten-year window."""

        if publication_year is None:
            return 0.0

        current_year = datetime.now().year
        age = max(0, current_year - publication_year)

        return max(0.0, 1.0 - age / 10)

    @staticmethod
    def _calculate_metadata_score(paper: Paper) -> float:
        """Measure completeness of basic bibliographic metadata."""

        fields = [
            bool(paper.title.strip()),
            bool(paper.authors),
            bool(paper.abstract and paper.abstract.strip()),
        ]

        return sum(fields) / len(fields)