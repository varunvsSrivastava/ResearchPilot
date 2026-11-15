
"""Tests for research prompt construction."""

import pytest

from researchpilot.rag.pipeline.prompt import ResearchPromptBuilder


def test_prompt_contains_query_and_context():
    """Prompt should contain the research question and evidence."""
    builder = ResearchPromptBuilder()

    prompt = builder.build(
        query="How does AI improve healthcare?",
        context="[Source 1]\nEvidence: AI supports diagnosis.",
    )

    assert "How does AI improve healthcare?" in prompt
    assert "[Source 1]" in prompt
    assert "AI supports diagnosis." in prompt


def test_prompt_contains_grounding_instructions():
    """Prompt should enforce evidence grounding and citations."""
    builder = ResearchPromptBuilder()

    prompt = builder.build(
        query="Research question",
        context="[Source 1]\nEvidence text.",
    )

    assert "using only the supplied evidence" in prompt
    assert "exact [Source N] label" in prompt
    assert "Never invent sources" in prompt
    assert "explicitly state what is missing" in prompt


def test_prompt_rejects_empty_query():
    """An empty research question should be rejected."""
    builder = ResearchPromptBuilder()

    with pytest.raises(ValueError, match="Query cannot be empty."):
        builder.build("   ", "Evidence text.")


def test_prompt_rejects_empty_context():
    """An empty evidence context should be rejected."""
    builder = ResearchPromptBuilder()

    with pytest.raises(ValueError, match="Context cannot be empty."):
        builder.build("Research question", "   ")
