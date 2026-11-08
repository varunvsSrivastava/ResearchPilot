
from unittest.mock import MagicMock, patch

import pytest

from researchpilot.planner.research_planner import (
    generate_research_plan,
)
from researchpilot.schemas.research_plan import ResearchPlan


@pytest.fixture
def mock_research_plan():
    return ResearchPlan(
        original_question="How is generative AI transforming healthcare?",
        research_objective="Analyze the impact of generative AI in healthcare.",
        sub_questions=[
            "What are its applications in healthcare?",
            "What challenges does it introduce?",
        ],
        search_queries=[
            "Generative AI applications in healthcare",
            "Generative AI healthcare challenges",
        ],
        source_types=["web", "academic"],
    )


def test_generate_research_plan_success(mock_research_plan):
    mock_response = MagicMock()
    mock_response.output_parsed = mock_research_plan

    with patch(
        "researchpilot.planner.research_planner.create_client"
    ) as mock_create_client:

        mock_client = MagicMock()
        mock_client.responses.parse.return_value = mock_response

        mock_create_client.return_value = mock_client

        result = generate_research_plan(
            "How is generative AI transforming healthcare?"
        )

    assert isinstance(result, ResearchPlan)
    assert result.original_question == (
        "How is generative AI transforming healthcare?"
    )
    assert len(result.sub_questions) == 2
    assert "web" in result.source_types
    assert "academic" in result.source_types


def test_generate_research_plan_empty_question():
    with pytest.raises(
        ValueError,
        match="Research question cannot be empty.",
    ):
        generate_research_plan("   ")


def test_generate_research_plan_missing_parsed_output():
    mock_response = MagicMock()
    mock_response.output_parsed = None

    with patch(
        "researchpilot.planner.research_planner.create_client"
    ) as mock_create_client:

        mock_client = MagicMock()
        mock_client.responses.parse.return_value = mock_response

        mock_create_client.return_value = mock_client

        with pytest.raises(
            RuntimeError,
            match="The model did not return a valid research plan.",
        ):
            generate_research_plan("Explain generative AI in healthcare.")


def test_generate_research_plan_api_error():
    with patch(
        "researchpilot.planner.research_planner.create_client"
    ) as mock_create_client:

        mock_client = MagicMock()
        mock_client.responses.parse.side_effect = RuntimeError(
            "Simulated API failure"
        )

        mock_create_client.return_value = mock_client

        with pytest.raises(RuntimeError, match="Simulated API failure"):
            generate_research_plan("Explain generative AI in healthcare.")


def test_generate_research_plan_unexpected_output_type():
    mock_response = MagicMock()
    mock_response.output_parsed = {"unexpected": "data"}

    with patch(
        "researchpilot.planner.research_planner.create_client"
    ) as mock_create_client:

        mock_client = MagicMock()
        mock_client.responses.parse.return_value = mock_response
        mock_create_client.return_value = mock_client

        with pytest.raises(
            RuntimeError,
            match="The model returned an unexpected research plan format.",
        ):
            generate_research_plan(
                "Explain generative AI in healthcare."
            )

def test_generate_research_plan_missing_output_attribute():
    mock_response = MagicMock(spec=[])

    with patch(
        "researchpilot.planner.research_planner.create_client"
    ) as mock_create_client:

        mock_client = MagicMock()
        mock_client.responses.parse.return_value = mock_response
        mock_create_client.return_value = mock_client

        with pytest.raises(
            RuntimeError,
            match="The model did not return a valid research plan.",
        ):
            generate_research_plan(
                "Explain generative AI in healthcare."
            )

