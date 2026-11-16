
"""Tests for evidence quality evaluation."""

import pytest

from researchpilot.evidence.quality import EvidenceQualityEvaluator


@pytest.fixture
def evaluator() -> EvidenceQualityEvaluator:
    return EvidenceQualityEvaluator()


def test_evaluate_returns_quality_score(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    content = (
        "Machine learning improves prediction accuracy. "
        "Models learn patterns from training data."
    )

    result = evaluator.evaluate(content)

    assert 0.0 <= result.completeness <= 1.0
    assert 0.0 <= result.clarity <= 1.0
    assert 0.0 <= result.overall_score <= 1.0


def test_evaluate_empty_content_raises_error(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    with pytest.raises(ValueError, match="Evidence content cannot be empty"):
        evaluator.evaluate("")


def test_evaluate_whitespace_content_raises_error(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    with pytest.raises(ValueError, match="Evidence content cannot be empty"):
        evaluator.evaluate("   ")


def test_completeness_increases_with_word_count(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    short_content = "Machine learning improves prediction."
    long_content = " ".join(["Machine"] * 100)

    short_score = evaluator.evaluate(short_content)
    long_score = evaluator.evaluate(long_content)

    assert long_score.completeness > short_score.completeness
    assert long_score.completeness == 1.0



def test_overall_score_is_average_of_dimensions(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    result = evaluator.evaluate(
        "Machine learning improves prediction accuracy."
    )

    expected = (
        result.relevance
        + result.completeness
        + result.clarity
    ) / 3

    assert result.overall_score == pytest.approx(expected)


def test_relevance_is_one_for_exact_query_term_coverage(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    result = evaluator.evaluate(
        content="Machine learning improves prediction.",
        research_question="machine learning prediction",
    )

    assert result.relevance == 1.0


def test_relevance_is_partial_for_partial_term_coverage(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    result = evaluator.evaluate(
        content="Machine learning improves prediction.",
        research_question="machine learning employee attrition",
    )

    assert result.relevance == pytest.approx(2 / 4)


def test_relevance_is_zero_without_matching_terms(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    result = evaluator.evaluate(
        content="Deep neural networks process images.",
        research_question="employee attrition prediction",
    )

    assert result.relevance == 0.0


def test_relevance_is_zero_for_empty_research_question(
    evaluator: EvidenceQualityEvaluator,
) -> None:
    result = evaluator.evaluate(
        content="Machine learning improves prediction.",
        research_question="",
    )

    assert result.relevance == 0.0
