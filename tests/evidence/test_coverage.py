
"""Tests for evidence coverage analysis."""

from researchpilot.evidence.coverage import EvidenceCoverageAnalyzer


def test_full_coverage():
    analyzer = EvidenceCoverageAnalyzer()

    result = analyzer.analyze(
        covered_aspects=["Methodology", "Results"],
        all_aspects=["Methodology", "Results"],
    )

    assert result.coverage_score == 1.0
    assert result.coverage_ratio == 1.0
    assert len(result.covered_aspects) == 2
    assert result.uncovered_aspects == []


def test_partial_coverage():
    analyzer = EvidenceCoverageAnalyzer()

    result = analyzer.analyze(
        covered_aspects=["Methodology"],
        all_aspects=["Methodology", "Results"],
    )

    assert result.coverage_score == 0.5
    assert result.coverage_ratio == 0.5
    assert result.covered_aspects == ["Methodology"]
    assert result.uncovered_aspects == ["Results"]


def test_zero_coverage():
    analyzer = EvidenceCoverageAnalyzer()

    result = analyzer.analyze(
        covered_aspects=[],
        all_aspects=["Methodology", "Results"],
    )

    assert result.coverage_score == 0.0
    assert result.covered_aspects == []
    assert len(result.uncovered_aspects) == 2


def test_empty_aspects():
    analyzer = EvidenceCoverageAnalyzer()

    result = analyzer.analyze(
        covered_aspects=[],
        all_aspects=[],
    )

    assert result.coverage_score == 0.0
    assert result.coverage_ratio == 0.0
    assert result.total_aspects == 0


def test_aspect_matching_is_case_insensitive():
    analyzer = EvidenceCoverageAnalyzer()

    result = analyzer.analyze(
        covered_aspects=["methodology"],
        all_aspects=["Methodology"],
    )

    assert result.covered_aspects == ["Methodology"]
    assert result.coverage_score == 1.0


def test_duplicate_aspects_are_not_counted_twice():
    analyzer = EvidenceCoverageAnalyzer()

    result = analyzer.analyze(
        covered_aspects=["Methodology"],
        all_aspects=["Methodology", "Methodology"],
    )

    assert result.total_aspects == 1
    assert result.coverage_score == 1.0
