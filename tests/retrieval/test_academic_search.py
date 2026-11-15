
from unittest.mock import Mock

import pytest

from researchpilot.retrieval.academic.base import AcademicSearchProvider
from researchpilot.retrieval.models import Paper
from researchpilot.retrieval.services.academic_search import (
    AcademicSearchService,
)
from researchpilot.schemas.research_plan import (
    ResearchPlan,
    SearchQuery,
    SourcePreferences,
)


@pytest.fixture
def research_plan() -> ResearchPlan:
    return ResearchPlan(
        original_question="How does AI improve healthcare?",
        research_objective="Investigate AI applications in healthcare.",
        sub_questions=[
            "How is AI used in diagnosis?",
            "What are AI healthcare limitations?",
        ],
        search_queries=[
            SearchQuery(
                query="AI healthcare diagnosis",
                purpose="Find research on diagnostic AI.",
                source_type="academic",
            ),
            SearchQuery(
                query="AI healthcare limitations",
                purpose="Find evidence about AI limitations.",
                source_type="academic",
            ),
            SearchQuery(
                query="AI healthcare news",
                purpose="Find supplementary web sources.",
                source_type="web",
            ),
        ],
        source_preferences=SourcePreferences(
            primary_sources=["academic"],
            preferred_evidence=["peer-reviewed research"],
        ),
        research_strategy="Review academic evidence and compare findings.",
    )


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        paper_id="W123",
        provider="openalex",
        title="AI in Healthcare",
        authors=["Author One"],
        source_url="https://openalex.org/W123",
        doi="10.1234/example",
        search_query="AI healthcare diagnosis",
    )


def test_search_plan_executes_academic_queries_only(
    research_plan: ResearchPlan,
    sample_paper: Paper,
) -> None:
    provider = Mock(spec=AcademicSearchProvider)
    provider.search.return_value = [sample_paper]

    service = AcademicSearchService(provider)
    results = service.search_plan(research_plan)

    assert len(results) == 2
    assert provider.search.call_count == 2
    assert results[0].query == "AI healthcare diagnosis"
    assert results[1].query == "AI healthcare limitations"


def test_search_plan_deduplicates_papers(
    research_plan: ResearchPlan,
    sample_paper: Paper,
) -> None:
    duplicate = sample_paper.model_copy(
        update={"paper_id": "W456", "title": "Different title"}
    )

    provider = Mock(spec=AcademicSearchProvider)
    provider.search.return_value = [sample_paper, duplicate]

    service = AcademicSearchService(provider)
    results = service.search_plan(research_plan)

    assert len(results[0].papers) == 1


def test_search_plan_rejects_invalid_limit(
    research_plan: ResearchPlan,
) -> None:
    provider = Mock(spec=AcademicSearchProvider)
    service = AcademicSearchService(provider)

    with pytest.raises(ValueError):
        service.search_plan(research_plan, max_results_per_query=0)


def test_search_plan_preserves_empty_results(
    research_plan: ResearchPlan,
) -> None:
    provider = Mock(spec=AcademicSearchProvider)
    provider.search.return_value = []

    service = AcademicSearchService(provider)
    results = service.search_plan(research_plan)

    assert len(results) == 2
    assert all(result.papers == [] for result in results)
