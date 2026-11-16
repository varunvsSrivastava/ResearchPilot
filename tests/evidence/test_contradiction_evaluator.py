
"""Tests for evidence contradiction assessment."""

from researchpilot.evidence.contradiction_evaluator import (
    ContradictionEvaluator,
)
from researchpilot.evidence.contradiction_models import (
    ContradictionAssessment,
)
from researchpilot.evidence.models import Evidence


def create_evidence(content: str, source_id: str) -> Evidence:
    """Create evidence for contradiction testing."""
    return Evidence(
        content=content,
        source_id=source_id,
        metadata={},
        retrieval_score=0.9,
    )


def test_evaluate_returns_assessment():
    evaluator = ContradictionEvaluator()

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

    result = evaluator.evaluate(evidence)

    assert isinstance(result, ContradictionAssessment)
    assert result.evidence_count == 2
    assert result.contradiction_count == 1
    assert len(result.contradictions) == 1


def test_evaluate_without_contradictions():
    evaluator = ContradictionEvaluator()

    evidence = [
        create_evidence("Research findings support the hypothesis.", "source-1"),
        create_evidence("Research findings support the hypothesis strongly.", "source-2"),
    ]

    result = evaluator.evaluate(evidence)

    assert result.evidence_count == 2
    assert result.contradiction_count == 0
    assert result.contradictions == []


def test_evaluate_empty_evidence():
    evaluator = ContradictionEvaluator()

    result = evaluator.evaluate([])

    assert result.evidence_count == 0
    assert result.contradiction_count == 0
    assert result.contradictions == []


def test_evaluate_multiple_contradictions():
    evaluator = ContradictionEvaluator()

    evidence = [
        create_evidence("The treatment improves patient survival.", "source-1"),
        create_evidence("The treatment does not improve patient survival.", "source-2"),
        create_evidence("The treatment improves patient survival outcomes.", "source-3"),
    ]

    result = evaluator.evaluate(evidence)

    assert result.evidence_count == 3
    assert result.contradiction_count >= 1
