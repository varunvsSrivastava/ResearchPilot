
from unittest.mock import MagicMock, patch

import pytest

from researchpilot.planner.research_planner import (
    generate_research_plan,
)
from researchpilot.schemas.research_plan import (
    ResearchPlan,
    SearchQuery,
    SourcePreferences,
)

@pytest.fixture
def mock_research_plan():
    return ResearchPlan(
        original_question="How is generative AI transforming healthcare?",
        research_objective=(
            "Analyze the impact of generative AI in healthcare."
        ),
        sub_questions=[
            "What are its applications in healthcare?",
            "What challenges does it introduce?",
        ],
        search_queries=[
            SearchQuery(
                query="Generative AI applications in healthcare",
                purpose="Identify healthcare applications.",
                source_type="academic",
            ),
            SearchQuery(
                query="Generative AI healthcare challenges",
                purpose="Identify implementation challenges.",
                source_type="web",
            ),
        ],
        source_preferences=SourcePreferences(
            primary_sources=["academic"],
            preferred_evidence=["Peer-reviewed research"],
            supplementary_sources=["web"],
        ),
        research_strategy=(
            "Prioritize academic research and supplement "
            "findings with reliable web sources."
        ),
    )


def test_generate_research_plan_success(mock_research_plan):
    question = "How is generative AI transforming healthcare?"

    mock_response = MagicMock()
    mock_response.output_parsed = mock_research_plan

    with (
        patch(
            "researchpilot.planner.research_planner.create_client"
        ) as mock_create_client,
        patch(
            "researchpilot.planner.research_planner.get_openai_model",
            return_value="test-model",
        ) as mock_get_model,
        patch(
            "researchpilot.planner.research_planner.track_usage"
        ) as mock_track_usage,
    ):
        mock_client = MagicMock()
        mock_client.responses.parse.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = generate_research_plan(question)

    # Validate returned research plan
    assert isinstance(result, ResearchPlan)
    assert result.original_question == question
    assert len(result.sub_questions) == 2
    assert "academic" in result.source_preferences.primary_sources
    assert "web" in result.source_preferences.supplementary_sources

    # Validate client and model configuration
    mock_create_client.assert_called_once()
    mock_get_model.assert_called_once()

    # Validate structured LLM request
    mock_client.responses.parse.assert_called_once()

    call_kwargs = mock_client.responses.parse.call_args.kwargs

    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["text_format"] == ResearchPlan

    # Validate system and user messages
    messages = call_kwargs["input"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == question

    # Validate usage tracking
    mock_track_usage.assert_called_once_with(mock_response)


def test_generate_research_plan_empty_question():
    with patch(
        "researchpilot.planner.research_planner.create_client"
    ) as mock_create_client:

        with pytest.raises(
            ValueError,
            match="Research question cannot be empty.",
        ):
            generate_research_plan("   ")

        mock_create_client.assert_not_called()


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

