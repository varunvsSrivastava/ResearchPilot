
"""Tests for research aspect coverage assessment."""

import pytest

from researchpilot.evidence.aspect_coverage import (
    AspectCoverageAssessor,
)


def test_assess_detects_covered_aspects():
    assessor = AspectCoverageAssessor()

    result = assessor.assess(
        aspects=["Employee attrition", "Salary analysis"],
        evidence_texts=[
            "Employee attrition is strongly associated with overtime."
        ],
    )

    assert result == ["Employee attrition"]


def test_assess_supports_partial_token_matching():
    assessor = AspectCoverageAssessor(minimum_match_ratio=0.5)

    result = assessor.assess(
        aspects=["Employee attrition"],
        evidence_texts=["Employee turnover increases over time."],
    )

    assert result == ["Employee attrition"]


def test_assess_returns_empty_without_evidence():
    assessor = AspectCoverageAssessor()

    result = assessor.assess(
        aspects=["Employee attrition"],
        evidence_texts=[],
    )

    assert result == []


def test_assess_ignores_aspects_without_meaningful_tokens():
    assessor = AspectCoverageAssessor()

    result = assessor.assess(
        aspects=["How is it"],
        evidence_texts=["How is it evaluated?"],
    )

    assert result == []


def test_invalid_match_ratio_raises_error():
    with pytest.raises(ValueError, match="Minimum match ratio"):
        AspectCoverageAssessor(minimum_match_ratio=0)
