
"""Unit tests for similarity ranking."""

import pytest

from researchpilot.rag.retrieval.ranking.similarity_ranker import (
    SimilarityRanker,
)
from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResult,
)


@pytest.fixture
def ranker() -> SimilarityRanker:
    """Provide a similarity ranker."""
    return SimilarityRanker()


def create_result(
    record_id: str,
    score: float,
) -> VectorSearchResult:
    """Create a vector search result for testing."""
    return VectorSearchResult(
        record=VectorRecord(
            id=record_id,
            text=f"Document {record_id}",
            vector=[0.1, 0.2],
        ),
        score=score,
    )


def test_rank_results_by_descending_score(
    ranker: SimilarityRanker,
) -> None:
    """Rank results from highest to lowest similarity."""
    results = [
        create_result("doc-1", 0.65),
        create_result("doc-2", 0.92),
        create_result("doc-3", 0.78),
    ]

    ranked_results = ranker.rank(results)

    assert [result.record.id for result in ranked_results] == [
        "doc-2",
        "doc-3",
        "doc-1",
    ]


def test_rank_empty_results(
    ranker: SimilarityRanker,
) -> None:
    """Return an empty list when no results are provided."""
    assert ranker.rank([]) == []


def test_rank_preserves_order_for_equal_scores(
    ranker: SimilarityRanker,
) -> None:
    """Preserve input order when similarity scores are equal."""
    results = [
        create_result("doc-1", 0.8),
        create_result("doc-2", 0.8),
        create_result("doc-3", 0.8),
    ]

    ranked_results = ranker.rank(results)

    assert [result.record.id for result in ranked_results] == [
        "doc-1",
        "doc-2",
        "doc-3",
    ]


def test_rank_does_not_modify_input(
    ranker: SimilarityRanker,
) -> None:
    """Ensure ranking does not mutate the original result list."""
    results = [
        create_result("doc-1", 0.4),
        create_result("doc-2", 0.9),
    ]

    original_results = results.copy()

    ranked_results = ranker.rank(results)

    assert results == original_results
    assert ranked_results is not results


def test_rank_single_result(
    ranker: SimilarityRanker,
) -> None:
    """Return the same result when only one candidate exists."""
    results = [create_result("doc-1", 0.75)]

    assert ranker.rank(results) == results
