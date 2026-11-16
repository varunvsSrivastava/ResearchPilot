
"""Tests for research evidence synthesis."""

from researchpilot.evidence.models import Evidence, EvidenceQualityScore, EvaluatedEvidence
from researchpilot.evidence.synthesis import ResearchSynthesizer
from researchpilot.evidence.synthesis_models import ResearchFinding


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


def test_synthesize_findings_returns_research_findings():
    synthesizer = ResearchSynthesizer()

    result = synthesizer.synthesize_findings(
        [create_evaluated_evidence("Research finding.", "source-1")]
    )

    assert len(result) == 1
    assert isinstance(result[0], ResearchFinding)
    assert result[0].finding == "Research finding."
    assert result[0].supporting_sources == ["source-1"]
    assert result[0].evidence_count == 1


def test_synthesize_findings_preserves_confidence():
    synthesizer = ResearchSynthesizer()

    result = synthesizer.synthesize_findings(
        [create_evaluated_evidence("Finding.", "source-1", 0.75)]
    )

    assert result[0].confidence_score == 0.75


def test_synthesize_multiple_evidence_items():
    synthesizer = ResearchSynthesizer()

    evidence = [
        create_evaluated_evidence("Finding one.", "source-1"),
        create_evaluated_evidence("Finding two.", "source-2"),
    ]

    result = synthesizer.synthesize_findings(evidence)

    assert len(result) == 2
    assert result[0].supporting_sources == ["source-1"]
    assert result[1].supporting_sources == ["source-2"]


def test_synthesize_empty_evidence():
    synthesizer = ResearchSynthesizer()

    result = synthesizer.synthesize_findings([])

    assert result == []
