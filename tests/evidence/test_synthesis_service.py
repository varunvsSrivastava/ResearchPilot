
"""Tests for research synthesis service."""

import pytest

from researchpilot.evidence.models import (
    Evidence,
    EvidenceQualityScore,
    EvaluatedEvidence,
)
from researchpilot.evidence.synthesis_models import ResearchSynthesis
from researchpilot.evidence.synthesis_service import ResearchSynthesisService


def create_evaluated_evidence(
    content: str,
    source_id: str,
    quality_score: float = 0.8,
) -> EvaluatedEvidence:
    """Create evaluated evidence for synthesis testing."""
    return EvaluatedEvidence(
        evidence=Evidence(
            content=content,
            source_id=source_id,
            metadata={},
            retrieval_score=0.9,
        ),
        quality=EvidenceQualityScore(
            relevance=quality_score,
            completeness=quality_score,
            clarity=quality_score,
            overall_score=quality_score,
        ),
    )


def test_synthesize_returns_research_synthesis():
    service = ResearchSynthesisService()

    result = service.synthesize(
        research_question="How does AI improve healthcare?",
        evaluated_evidence=[
            create_evaluated_evidence("AI improves diagnosis.", "source-1")
        ],
    )

    assert isinstance(result, ResearchSynthesis)
    assert result.research_question == "How does AI improve healthcare?"
    assert result.evidence_count == 1
    assert len(result.findings) == 1


def test_synthesize_empty_evidence():
    service = ResearchSynthesisService()

    result = service.synthesize(
        research_question="Research question",
        evaluated_evidence=[],
    )

    assert result.evidence_count == 0
    assert result.findings == []
    assert result.synthesis_summary


def test_synthesis_summary_contains_counts():
    service = ResearchSynthesisService()

    result = service.synthesize(
        research_question="Research question",
        evaluated_evidence=[
            create_evaluated_evidence("Finding one.", "source-1"),
            create_evaluated_evidence("Finding two.", "source-2"),
        ],
    )

    assert "2 findings" in result.synthesis_summary
    assert "2 evidence items" in result.synthesis_summary


def test_empty_research_question_raises_error():
    service = ResearchSynthesisService()

    with pytest.raises(ValueError, match="Research question"):
        service.synthesize(
            research_question=" ",
            evaluated_evidence=[],
        )
