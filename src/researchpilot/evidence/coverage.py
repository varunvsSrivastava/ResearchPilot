
"""Evidence coverage analysis utilities."""

from researchpilot.evidence.coverage_models import (
    EvidenceCoverageScore,
)


class EvidenceCoverageAnalyzer:
    """Analyze evidence coverage across research-question aspects."""

    def analyze(
        self,
        covered_aspects: list[str],
        all_aspects: list[str],
    ) -> EvidenceCoverageScore:
        """Calculate coverage based on covered research aspects."""

        normalized_covered = {
            aspect.strip().lower()
            for aspect in covered_aspects
            if aspect.strip()
        }

        normalized_all = {
            aspect.strip().lower(): aspect
            for aspect in all_aspects
            if aspect.strip()
        }

        covered = [
            aspect
            for normalized, aspect in normalized_all.items()
            if normalized in normalized_covered
        ]

        uncovered = [
            aspect
            for normalized, aspect in normalized_all.items()
            if normalized not in normalized_covered
        ]

        total_aspects = len(normalized_all)
        coverage_ratio = (
            len(covered) / total_aspects
            if total_aspects
            else 0.0
        )

        return EvidenceCoverageScore(
            coverage_score=coverage_ratio,
            covered_aspects=covered,
            uncovered_aspects=uncovered,
            total_aspects=total_aspects,
            coverage_ratio=coverage_ratio,
        )
