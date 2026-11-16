
"""Tests for evidence contradiction detection."""

from researchpilot.evidence.contradiction import ContradictionDetector
from researchpilot.evidence.models import Evidence


def create_evidence(
    content: str,
    source_id: str,
) -> Evidence:
    """Create evidence for contradiction testing."""
    return Evidence(
        content=content,
        source_id=source_id,
        metadata={},
        retrieval_score=0.9,
    )


def test_detects_negation_contradiction():
    detector = ContradictionDetector()

    evidence = [
        create_evidence(
            "The treatment improves patient survival.",
            "source-1",
        ),
        create_evidence(
            "The treatment does not improve patient survival.",
            "source-2",
        ),
    ]

    result = detector.detect(evidence)

    assert len(result) == 1
    assert result[0].first_source_id == "source-1"
    assert result[0].second_source_id == "source-2"
    assert result[0].contradiction_score > 0.0


def test_no_contradiction_for_similar_positive_claims():
    detector = ContradictionDetector()

    evidence = [
        create_evidence(
            "The treatment improves patient survival.",
            "source-1",
        ),
        create_evidence(
            "The treatment improves patient survival significantly.",
            "source-2",
        ),
    ]

    result = detector.detect(evidence)

    assert result == []


def test_no_contradiction_for_unrelated_claims():
    detector = ContradictionDetector()

    evidence = [
        create_evidence(
            "The treatment improves patient survival.",
            "source-1",
        ),
        create_evidence(
            "Climate change affects global temperatures.",
            "source-2",
        ),
    ]

    result = detector.detect(evidence)

    assert result == []


def test_detect_returns_empty_for_single_evidence():
    detector = ContradictionDetector()

    result = detector.detect(
        [create_evidence("Research evidence statement.", "source-1")]
    )

    assert result == []


def test_detect_returns_empty_for_no_evidence():
    detector = ContradictionDetector()

    result = detector.detect([])

    assert result == []
