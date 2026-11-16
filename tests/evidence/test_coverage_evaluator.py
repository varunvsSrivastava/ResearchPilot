
"""Tests for evidence coverage assessment."""

import pytest

from researchpilot.evidence.coverage_evaluator import (
    EvidenceCoverageEvaluator,
)
from researchpilot.evidence.coverage_models import (
    EvidenceCoverageAssessment,
)


def test_evaluate_returns_assessment():
    evaluator = EvidenceCoverageEvaluator()

    result = evaluator.evaluate(
        research_question="How does AI improve healthcare?",
        evidence_count=10,
        covered_aspects=["Accuracy", "Efficiency"],
        all_aspects=["Accuracy", "Efficiency", "Cost"],
    )

    assert isinstance(result, EvidenceCoverageAssessment)
    assert result.research_question == "How does AI improve healthcare?"
    assert result.evidence_count == 10
    assert result.coverage.coverage_score == pytest.approx(2 / 3)
    assert result.coverage.uncovered_aspects == ["Cost"]


def test_evaluate_full_coverage():
    evaluator = EvidenceCoverageEvaluator()

    result = evaluator.evaluate(
        research_question="Research question",
        evidence_count=5,
        covered_aspects=["A", "B"],
        all_aspects=["A", "B"],
    )

    assert result.coverage.coverage_score == 1.0


def test_evaluate_empty_aspects():
    evaluator = EvidenceCoverageEvaluator()

    result = evaluator.evaluate(
        research_question="Research question",
        evidence_count=0,
        covered_aspects=[],
        all_aspects=[],
    )

    assert result.coverage.coverage_score == 0.0
    assert result.coverage.total_aspects == 0


def test_empty_research_question_raises_error():
    evaluator = EvidenceCoverageEvaluator()

    with pytest.raises(ValueError, match="Research question"):
        evaluator.evaluate(
            research_question=" ",
            evidence_count=0,
            covered_aspects=[],
            all_aspects=[],
        )


def test_negative_evidence_count_raises_error():
    evaluator = EvidenceCoverageEvaluator()

    with pytest.raises(ValueError, match="Evidence count"):
        evaluator.evaluate(
            research_question="Research question",
            evidence_count=-1,
            covered_aspects=[],
            all_aspects=[],
        )
