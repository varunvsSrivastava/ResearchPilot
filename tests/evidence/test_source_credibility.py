"""Tests for academic source credibility evaluation."""

from researchpilot.evidence.source_credibility import (
    SourceCredibilityEvaluator,
)
from researchpilot.retrieval.models import Paper


def create_paper(**overrides) -> Paper:
    """Create a paper with default metadata for testing."""
    data = {
        "paper_id": "W123",
        "provider": "openalex",
        "title": "Research Paper",
        "abstract": "A research abstract.",
        "authors": ["Author One"],
        "publication_year": 2025,
        "doi": "10.1234/example",
        "source_url": "https://openalex.org/W123",
        "pdf_url": None,
        "pdf_urls": [],
        "citation_count": 100,
        "is_open_access": True,
        "search_query": "research",
    }
    data.update(overrides)
    return Paper(**data)


def test_evaluate_returns_all_scores():
    evaluator = SourceCredibilityEvaluator()

    result = evaluator.evaluate(create_paper())

    assert result.doi_score == 1.0
    assert result.citation_score > 0.0
    assert result.recency_score > 0.0
    assert result.metadata_score == 1.0
    assert 0.0 <= result.overall_score <= 1.0


def test_missing_doi_scores_zero():
    evaluator = SourceCredibilityEvaluator()

    result = evaluator.evaluate(create_paper(doi=None))

    assert result.doi_score == 0.0


def test_missing_citations_scores_zero():
    evaluator = SourceCredibilityEvaluator()

    result = evaluator.evaluate(create_paper(citation_count=None))

    assert result.citation_score == 0.0


def test_zero_citations_scores_zero():
    evaluator = SourceCredibilityEvaluator()

    result = evaluator.evaluate(create_paper(citation_count=0))

    assert result.citation_score == 0.0


def test_missing_publication_year_scores_zero():
    evaluator = SourceCredibilityEvaluator()

    result = evaluator.evaluate(create_paper(publication_year=None))

    assert result.recency_score == 0.0


def test_old_publication_has_zero_recency():
    evaluator = SourceCredibilityEvaluator()

    result = evaluator.evaluate(create_paper(publication_year=2000))

    assert result.recency_score == 0.0


def test_missing_metadata_reduces_metadata_score():
    evaluator = SourceCredibilityEvaluator()

    paper = create_paper(
        authors=[],
        abstract=None,
    )

    result = evaluator.evaluate(paper)

    assert result.metadata_score == 1 / 3