
"""Evidence coverage assessment service."""

from researchpilot.evidence.coverage import EvidenceCoverageAnalyzer
from researchpilot.evidence.coverage_models import (
    EvidenceCoverageAssessment,
)


class EvidenceCoverageEvaluator:
    """Evaluate evidence coverage for a research question."""

    def __init__(self) -> None:
        self._coverage_analyzer = EvidenceCoverageAnalyzer()

    def evaluate(
        self,
        research_question: str,
        evidence_count: int,
        covered_aspects: list[str],
        all_aspects: list[str],
    ) -> EvidenceCoverageAssessment:
        """Generate a complete evidence coverage assessment."""

        if not research_question.strip():
            raise ValueError("Research question must not be empty.")

        if evidence_count < 0:
            raise ValueError("Evidence count must not be negative.")

        coverage = self._coverage_analyzer.analyze(
            covered_aspects=covered_aspects,
            all_aspects=all_aspects,
        )

        return EvidenceCoverageAssessment(
            research_question=research_question,
            evidence_count=evidence_count,
            coverage=coverage,
        )
