
"""Tests for structured report citation validation."""

import pytest

from researchpilot.generation.citation_validator import (
    ReportCitationValidator,
)
from researchpilot.generation.report_models import (
    ResearchReport,
    ReportSection,
)


@pytest.fixture
def validator():
    return ReportCitationValidator()


@pytest.fixture
def context():
    return (
        "[Source 1] Evidence one\n"
        "[Source 2] Evidence two\n"
        "[Source 3] Evidence three"
    )


def test_valid_report_citations(validator, context):
    report = ResearchReport(
        title="Test Report",
        research_question="Test question",
        sections=[
            ReportSection(
                title="Finding 1",
                content="Finding supported by [Source 1].",
                citations=[1],
            ),
            ReportSection(
                title="Finding 2",
                content="Finding supported by [Source 2].",
                citations=[2],
            ),
        ],
        conclusion="Conclusion supported by [Source 3].",
    )

    result = validator.validate(report, context)

    assert result.is_valid
    assert result.cited_sources == {1, 2, 3}
    assert result.invalid_sources == set()
    assert result.missing_citations == []
    assert result.unused_sources == set()


def test_invalid_source_reference(validator, context):
    report = ResearchReport(
        title="Test Report",
        research_question="Test question",
        sections=[
            ReportSection(
                title="Finding 1",
                content="Finding supported by [Source 5].",
                citations=[5],
            )
        ],
    )

    result = validator.validate(report, context)

    assert not result.is_valid
    assert result.invalid_sources == {5}


def test_missing_section_citation(validator, context):
    report = ResearchReport(
        title="Test Report",
        research_question="Test question",
        sections=[
            ReportSection(
                title="Finding 1",
                content="Unsupported finding.",
            )
        ],
    )

    result = validator.validate(report, context)

    assert not result.is_valid
    assert result.missing_citations == ["Finding 1"]


def test_missing_conclusion_citation(validator, context):
    report = ResearchReport(
        title="Test Report",
        research_question="Test question",
        conclusion="Unsupported conclusion.",
    )

    result = validator.validate(report, context)

    assert not result.is_valid
    assert result.missing_citations == ["Conclusion"]


def test_unused_context_sources(validator, context):
    report = ResearchReport(
        title="Test Report",
        research_question="Test question",
        sections=[
            ReportSection(
                title="Finding 1",
                content="Finding supported by [Source 1].",
                citations=[1],
            )
        ],
    )

    result = validator.validate(report, context)

    assert result.is_valid
    assert result.unused_sources == {2, 3}


def test_empty_report(validator):
    report = ResearchReport(
        title="Empty Report",
        research_question="Test question",
    )

    result = validator.validate(report, context="")

    assert result.is_valid
    assert result.cited_sources == set()
    assert result.invalid_sources == set()
    assert result.missing_citations == []
    assert result.unused_sources == set()
