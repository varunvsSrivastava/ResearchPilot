
"""Tests for evidence-grounded answer generation."""

from unittest.mock import patch

import pytest

from researchpilot.rag.pipeline.generation import (
    AnswerGenerationService,
)


@patch("researchpilot.rag.pipeline.generation.generate_response")
def test_generate_answer_success(mock_generate_response):
    """Generation service should return the LLM response."""
    mock_generate_response.return_value = (
        "AI improves healthcare through diagnostic support. [Source 1]"
    )

    service = AnswerGenerationService()

    result = service.generate(
        query="How does AI improve healthcare?",
        context="[Source 1]\nEvidence: AI supports diagnosis.",
    )

    assert result == (
        "AI improves healthcare through diagnostic support. [Source 1]"
    )

    mock_generate_response.assert_called_once()

    prompt = mock_generate_response.call_args.args[0]

    assert "How does AI improve healthcare?" in prompt
    assert "[Source 1]" in prompt


@patch("researchpilot.rag.pipeline.generation.generate_response")
def test_generate_answer_propagates_llm_error(mock_generate_response):
    """LLM errors should propagate to the caller."""
    mock_generate_response.side_effect = RuntimeError(
        "LLM generation failed."
    )

    service = AnswerGenerationService()

    with pytest.raises(
        RuntimeError,
        match="LLM generation failed.",
    ):
        service.generate(
            query="Research question",
            context="[Source 1]\nEvidence text.",
        )


def test_generate_answer_rejects_empty_query():
    """Empty research questions should be rejected."""
    service = AnswerGenerationService()

    with pytest.raises(ValueError, match="Query cannot be empty."):
        service.generate(
            query="   ",
            context="Evidence text.",
        )

@patch("researchpilot.rag.pipeline.generation.generate_response")
def test_generate_answer_with_empty_context(
    mock_generate_response,
):
    """Empty evidence should produce a safe fallback."""
    service = AnswerGenerationService()

    result = service.generate(
        query="What are the effects of AI?",
        context="",
    )

    assert result == service.INSUFFICIENT_EVIDENCE_RESPONSE
    mock_generate_response.assert_not_called()


@patch("researchpilot.rag.pipeline.generation.generate_response")
def test_generate_answer_with_whitespace_context(
    mock_generate_response,
):
    """Whitespace evidence should produce a safe fallback."""
    service = AnswerGenerationService()

    result = service.generate(
        query="What are the effects of AI?",
        context="   \n\t",
    )

    assert result == service.INSUFFICIENT_EVIDENCE_RESPONSE
    mock_generate_response.assert_not_called()


@patch("researchpilot.rag.pipeline.generation.generate_response")
def test_generate_answer_rejects_unknown_citations(
    mock_generate_response,
):
    """Answers citing unavailable sources should be rejected."""
    mock_generate_response.return_value = (
        "AI improves diagnosis [Source 3]."
    )

    service = AnswerGenerationService()

    result = service.generate(
        query="How does AI improve healthcare?",
        context="[Source 1]\nEvidence: AI supports diagnosis.",
    )

    assert result == service.INVALID_CITATION_RESPONSE


@patch("researchpilot.rag.pipeline.generation.generate_response")
def test_generate_answer_accepts_valid_citations(
    mock_generate_response,
):
    """Answers citing available sources should be accepted."""
    mock_generate_response.return_value = (
        "AI improves diagnosis [Source 1]."
    )

    service = AnswerGenerationService()

    result = service.generate(
        query="How does AI improve healthcare?",
        context="[Source 1]\nEvidence: AI supports diagnosis.",
    )

    assert result == "AI improves diagnosis [Source 1]."
