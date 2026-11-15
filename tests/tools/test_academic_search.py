
"""Tests for academic search tool."""

import pytest
from pydantic import ValidationError

from researchpilot.tools.academic_search import AcademicSearchInput


def test_valid_academic_search_input():
    arguments = AcademicSearchInput(
        queries=["retrieval augmented generation", "RAG evaluation"],
        max_results_per_query=5,
    )

    assert arguments.queries == [
        "retrieval augmented generation",
        "RAG evaluation",
    ]
    assert arguments.max_results_per_query == 5


def test_default_max_results():
    arguments = AcademicSearchInput(
        queries=["large language models", "LLM evaluation"],
    )

    assert arguments.max_results_per_query == 10


def test_empty_queries_rejected():
    with pytest.raises(ValidationError):
        AcademicSearchInput(queries=[])


def test_invalid_max_results_rejected():
    with pytest.raises(ValidationError):
        AcademicSearchInput(
            queries=["machine learning"],
            max_results_per_query=100,
        )


def test_multiple_queries_supported():
    arguments = AcademicSearchInput(
        queries=["query one", "query two"],
    )

    assert len(arguments.queries) == 2


from unittest.mock import MagicMock

from researchpilot.tools.academic_search import (
    AcademicSearchInput,
    AcademicSearchTool,
)


@pytest.fixture
def academic_search_service():
    service = MagicMock()
    service.search_plan.return_value = ["paper_1", "paper_2"]
    return service


@pytest.fixture
def academic_tool(academic_search_service):
    return AcademicSearchTool(academic_search_service)


def test_academic_search_executes(academic_tool):
    arguments = AcademicSearchInput(
        queries=["retrieval augmented generation", "RAG evaluation"]
    )

    results = academic_tool.execute(arguments)

    assert results == ["paper_1", "paper_2"]


def test_search_service_called_once(
    academic_tool,
    academic_search_service,
):
    arguments = AcademicSearchInput(
        queries=["large language models", "LLM evaluation"]
    )

    academic_tool.execute(arguments)

    academic_search_service.search_plan.assert_called_once()


def test_max_results_forwarded(
    academic_tool,
    academic_search_service,
):
    arguments = AcademicSearchInput(
        queries=["machine learning", "deep learning"],
        max_results_per_query=5,
    )

    academic_tool.execute(arguments)

    _, kwargs = academic_search_service.search_plan.call_args
    assert kwargs["max_results_per_query"] == 5


def test_multiple_queries_forwarded(
    academic_tool,
    academic_search_service,
):
    arguments = AcademicSearchInput(
        queries=["query one", "query two"]
    )

    academic_tool.execute(arguments)

    research_plan = (
        academic_search_service.search_plan.call_args.args[0]
    )

    assert len(research_plan.search_queries) == 2
    assert all(
        query.source_type == "academic"
        for query in research_plan.search_queries
    )


def test_results_are_returned_unchanged(
    academic_tool,
    academic_search_service,
):
    expected_results = ["paper_a", "paper_b"]
    academic_search_service.search_plan.return_value = expected_results

    results = academic_tool.execute(
        AcademicSearchInput(queries=["AI research", "AI evaluation"])
    )

    assert results is expected_results

def test_single_query_rejected():
    with pytest.raises(ValidationError):
        AcademicSearchInput(queries=["machine learning"])