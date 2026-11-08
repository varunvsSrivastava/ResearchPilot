
from unittest.mock import MagicMock, patch

import pytest

from researchpilot.llm.client import generate_response
from researchpilot.llm.reliability import (
    LLMEmptyResponseError,
    LLMUnexpectedResponseError,
)


@patch("researchpilot.llm.client.get_openai_model")
@patch("researchpilot.llm.client.create_client")
def test_generate_response_success(
    mock_create_client,
    mock_get_model,
):
    mock_get_model.return_value = "gpt-4o-mini"

    mock_client = MagicMock()
    mock_client.responses.create.return_value.output_text = (
        "ResearchPilot response successful."
    )
    mock_create_client.return_value = mock_client

    result = generate_response("Explain generative AI.")

    assert result == "ResearchPilot response successful."

    mock_client.responses.create.assert_called_once()


@pytest.mark.parametrize("output_text", ["", "   ", "\n\t"])
@patch("researchpilot.llm.client.create_client")
def test_generate_response_empty_output(
    mock_create_client,
    output_text,
):
    mock_client = MagicMock()
    mock_client.responses.create.return_value.output_text = output_text
    mock_create_client.return_value = mock_client

    with pytest.raises(
        LLMEmptyResponseError,
        match="The LLM API returned an empty response.",
    ):
        generate_response("Explain generative AI.")


@pytest.mark.parametrize("output_text", [None, 123, []])
@patch("researchpilot.llm.client.create_client")
def test_generate_response_unexpected_output(
    mock_create_client,
    output_text,
):
    mock_client = MagicMock()
    mock_client.responses.create.return_value.output_text = output_text
    mock_create_client.return_value = mock_client

    with pytest.raises(LLMUnexpectedResponseError):
        generate_response("Explain generative AI.")


@patch("researchpilot.llm.client.create_client")
def test_generate_response_missing_output_text(
    mock_create_client,
):
    mock_client = MagicMock()
    mock_response = MagicMock(spec=[])
    mock_client.responses.create.return_value = mock_response
    mock_create_client.return_value = mock_client

    with pytest.raises(LLMUnexpectedResponseError):
        generate_response("Explain generative AI.")


def test_generate_response_empty_prompt():
    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty.",
    ):
        generate_response("   ")
