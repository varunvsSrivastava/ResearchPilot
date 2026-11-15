
"""Tests for citation validation."""

from researchpilot.rag.pipeline.citations import CitationValidator


def test_extract_citations():
    validator = CitationValidator()

    answer = "Finding one [Source 1]. Finding two [Source 2]."

    assert validator.extract_citations(answer) == {1, 2}


def test_extract_available_sources():
    validator = CitationValidator()

    context = (
        "[Source 1]\nEvidence one.\n\n"
        "[Source 2]\nEvidence two."
    )

    assert validator.extract_available_sources(context) == {1, 2}


def test_validate_citations_success():
    validator = CitationValidator()

    answer = "AI improves diagnosis [Source 1]."
    context = "[Source 1]\nEvidence: AI supports diagnosis."

    assert validator.validate(answer, context) is True


def test_validate_unknown_citation():
    validator = CitationValidator()

    answer = "AI improves diagnosis [Source 3]."
    context = "[Source 1]\nEvidence: AI supports diagnosis."

    assert validator.validate(answer, context) is False


def test_validate_answer_without_citations():
    validator = CitationValidator()

    answer = "AI improves diagnosis."
    context = "[Source 1]\nEvidence: AI supports diagnosis."

    assert validator.validate(answer, context) is True
