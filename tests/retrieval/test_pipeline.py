
"""Tests for the integrated retrieval pipeline."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from researchpilot.retrieval.fulltext.models import Evidence
from researchpilot.retrieval.fulltext.service import (
    FullTextAcquisitionError,
)
from researchpilot.retrieval.models import Paper, SearchResult
from researchpilot.retrieval.pipeline import RetrievalPipeline
from researchpilot.schemas.research_plan import (
    ResearchPlan,
    SearchQuery,
    SourcePreferences,
)


@pytest.fixture
def research_plan() -> ResearchPlan:
    """Create a sample research plan."""
    return ResearchPlan(
        original_question="How does RAG improve LLM reliability?",
        research_objective="Evaluate the impact of RAG on LLM reliability.",
        sub_questions=[
            "How does RAG reduce hallucinations?",
            "What are RAG limitations?",
        ],
        search_queries=[
            SearchQuery(
                query="RAG reliability",
                purpose="Find research on RAG reliability.",
                source_type="academic",
            ),
            SearchQuery(
                query="RAG hallucination",
                purpose="Find research on RAG hallucinations.",
                source_type="academic",
            ),
        ],
        source_preferences=SourcePreferences(
            primary_sources=["academic"],
            preferred_evidence=["peer-reviewed research"],
        ),
        research_strategy=(
            "Review academic evidence on RAG reliability "
            "and hallucination reduction."
        ),
    )


@pytest.fixture
def paper_factory():
    """Create sample academic papers."""
    def create_paper(
        paper_id: str,
        title: str,
        doi: str | None = None,
    ) -> Paper:
        return Paper(
            paper_id=paper_id,
            provider="openalex",
            title=title,
            authors=["Author One"],
            publication_year=2024,
            doi=doi,
            source_url=f"https://example.com/{paper_id}",
            pdf_url=f"https://example.com/{paper_id}.pdf",
            search_query="RAG reliability",
        )

    return create_paper



@pytest.fixture
def evidence_factory():
    """Create sample evidence records."""

    def create_evidence(
        evidence_id: str,
        paper_id: str,
    ) -> Evidence:
        return Evidence(
            evidence_id=evidence_id,
            paper_id=paper_id,
            title=f"Evidence title for {paper_id}",
            text=f"Evidence text for {paper_id}.",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=30,
            source_url=f"https://example.com/{paper_id}",
            pdf_url=f"https://example.com/{paper_id}.pdf",
        )

    return create_evidence



def test_execute_collects_evidence(
    research_plan,
    paper_factory,
    evidence_factory,
):
    """Collect evidence from successfully acquired papers."""
    paper_one = paper_factory("paper-1", "RAG Reliability")
    paper_two = paper_factory("paper-2", "RAG Hallucinations")

    search_service = Mock()
    search_service.search_plan.return_value = [
        SearchResult(
            query="RAG reliability",
            provider="openalex",
            papers=[paper_one, paper_two],
            total_results=2,
        )
    ]

    fulltext_service = Mock()
    fulltext_service.acquire.side_effect = [
        [evidence_factory("ev-1", "paper-1")],
        [evidence_factory("ev-2", "paper-2")],
    ]

    pipeline = RetrievalPipeline(search_service, fulltext_service)

    result = pipeline.execute(research_plan)

    assert result.research_question == research_plan.original_question
    assert result.total_papers == 2
    assert result.successful_papers == 2
    assert result.failed_papers == 0
    assert result.total_evidence == 2
    assert len(result.evidence) == 2
    assert len(result.paper_results) == 2


def test_execute_deduplicates_papers(
    research_plan,
    paper_factory,
    evidence_factory,
):
    """Deduplicate papers across academic search results."""
    paper_one = paper_factory(
        "paper-1",
        "RAG Reliability",
        "10.1234/example",
    )
    duplicate = paper_factory(
        "paper-duplicate",
        "Different Title",
        "10.1234/example",
    )
    paper_two = paper_factory("paper-2", "RAG Hallucinations")

    search_service = Mock()
    search_service.search_plan.return_value = [
        SearchResult(
            query="query-1",
            provider="openalex",
            papers=[paper_one, duplicate],
            total_results=2,
        ),
        SearchResult(
            query="query-2",
            provider="openalex",
            papers=[paper_one, paper_two],
            total_results=2,
        ),
    ]

    fulltext_service = Mock()
    fulltext_service.acquire.side_effect = [
        [evidence_factory("ev-1", "paper-1")],
        [evidence_factory("ev-2", "paper-2")],
    ]

    pipeline = RetrievalPipeline(search_service, fulltext_service)

    result = pipeline.execute(research_plan)

    assert result.total_papers == 2
    assert fulltext_service.acquire.call_count == 2
    assert [paper.paper_id for paper in result.papers] == [
        "paper-1",
        "paper-2",
    ]


def test_execute_continues_after_acquisition_failure(
    research_plan,
    paper_factory,
    evidence_factory,
):
    """Continue processing papers when one acquisition fails."""
    paper_one = paper_factory("paper-1", "RAG Reliability")
    paper_two = paper_factory("paper-2", "RAG Hallucinations")

    search_service = Mock()
    search_service.search_plan.return_value = [
        SearchResult(
            query="RAG reliability",
            provider="openalex",
            papers=[paper_one, paper_two],
            total_results=2,
        )
    ]

    fulltext_service = Mock()
    fulltext_service.acquire.side_effect = [
        FullTextAcquisitionError("PDF unavailable"),
        [evidence_factory("ev-2", "paper-2")],
    ]

    pipeline = RetrievalPipeline(search_service, fulltext_service)

    result = pipeline.execute(research_plan)

    assert result.total_papers == 2
    assert result.successful_papers == 1
    assert result.failed_papers == 1
    assert result.total_evidence == 1
    assert result.paper_results[0].error == "PDF unavailable"
    assert result.paper_results[1].success is True


def test_execute_handles_empty_search_results(research_plan):
    """Return empty retrieval output when no papers are found."""
    search_service = Mock()
    search_service.search_plan.return_value = []

    fulltext_service = Mock()

    pipeline = RetrievalPipeline(search_service, fulltext_service)

    result = pipeline.execute(research_plan)

    assert result.total_papers == 0
    assert result.successful_papers == 0
    assert result.failed_papers == 0
    assert result.total_evidence == 0
    assert result.evidence == []
    fulltext_service.acquire.assert_not_called()


def test_execute_passes_search_limit(research_plan):
    """Forward the configured search limit to academic search."""
    search_service = Mock()
    search_service.search_plan.return_value = []

    pipeline = RetrievalPipeline(search_service, Mock())

    pipeline.execute(
        research_plan,
        max_results_per_query=5,
    )

    search_service.search_plan.assert_called_once_with(
        research_plan=research_plan,
        max_results_per_query=5,
    )
