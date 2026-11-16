
"""Tests for structured research report generation."""

import pytest

from researchpilot.evidence.synthesis_models import (
    ResearchFinding,
    ResearchSynthesis,
)
from researchpilot.generation.report_generator import ReportGenerator


@pytest.fixture
def generator():
    return ReportGenerator()


@pytest.fixture
def synthesis():
    return ResearchSynthesis(
        research_question="How does AI improve healthcare?",
        findings=[
            ResearchFinding(
                finding="AI improves diagnostic accuracy.",
                supporting_sources=["[Source 1]", "[Source 2]"],
                evidence_count=2,
                confidence_score=0.9,
            ),
            ResearchFinding(
                finding="AI can reduce clinical workload.",
                supporting_sources=["[Source 3]"],
                evidence_count=1,
                confidence_score=0.8,
            ),
        ],
        evidence_count=3,
        synthesis_summary="AI offers benefits in healthcare.",
    )


def test_generate_report(generator, synthesis):
    context = "[Source 1] Evidence A\n[Source 2] Evidence B\n[Source 3] Evidence C"

    report = generator.generate(synthesis, context)

    assert report.title == "Research Report"
    assert report.research_question == synthesis.research_question
    assert len(report.sections) == 2
    assert report.sections[0].title == "Finding 1"
    assert report.sections[0].content == "AI improves diagnostic accuracy."
    assert report.sections[0].citations == [1, 2]
    assert report.sections[1].citations == [3]
    assert report.conclusion == synthesis.synthesis_summary
    assert report.metadata["evidence_count"] == "3"
    assert report.metadata["finding_count"] == "2"


def test_generate_report_with_custom_title(generator, synthesis):
    context = "[Source 1] Evidence A\n[Source 2] Evidence B\n[Source 3] Evidence C"

    report = generator.generate(
        synthesis,
        context,
        title="AI in Healthcare",
    )

    assert report.title == "AI in Healthcare"


def test_reject_invalid_citations(generator, synthesis):
    context = "[Source 1] Evidence A\n[Source 2] Evidence B"

    with pytest.raises(ValueError, match="Report contains citations"):
        generator.generate(synthesis, context)


def test_generate_report_without_findings(generator):
    synthesis = ResearchSynthesis(
        research_question="Research question",
        findings=[],
        evidence_count=0,
        synthesis_summary="No findings available.",
    )

    report = generator.generate(synthesis, context="")

    assert report.sections == []
    assert report.conclusion == "No findings available."
    assert report.metadata["finding_count"] == "0"


def test_extract_citations_preserves_order(generator):
    citations = generator._extract_citations(
        "[Source 2] [Source 1] [Source 2] [Source 3]"
    )

    assert citations == [2, 1, 3]
