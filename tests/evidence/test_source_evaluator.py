
"""Tests for academic source evaluation."""

from researchpilot.evidence.source_evaluator import SourceEvaluator
from researchpilot.evidence.source_models import EvaluatedSource
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


def test_evaluate_returns_evaluated_sources():
    evaluator = SourceEvaluator()

    result = evaluator.evaluate([create_paper()])

    assert len(result) == 1
    assert isinstance(result[0], EvaluatedSource)
    assert result[0].paper_id == "W123"
    assert result[0].title == "Research Paper"
    assert result[0].provider == "openalex"


def test_evaluate_multiple_papers():
    evaluator = SourceEvaluator()

    papers = [
        create_paper(paper_id="W123"),
        create_paper(paper_id="W456", title="Second Paper"),
    ]

    result = evaluator.evaluate(papers)

    assert len(result) == 2
    assert result[0].paper_id == "W123"
    assert result[1].paper_id == "W456"
    assert result[1].title == "Second Paper"


def test_evaluate_preserves_credibility_scores():
    evaluator = SourceEvaluator()

    result = evaluator.evaluate([create_paper()])

    credibility = result[0].credibility

    assert credibility.doi_score == 1.0
    assert credibility.metadata_score == 1.0
    assert 0.0 <= credibility.overall_score <= 1.0


def test_evaluate_empty_papers_returns_empty_list():
    evaluator = SourceEvaluator()

    result = evaluator.evaluate([])

    assert result == []
