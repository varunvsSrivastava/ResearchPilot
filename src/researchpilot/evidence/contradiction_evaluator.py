
"""Evidence contradiction assessment service."""

from researchpilot.evidence.contradiction import ContradictionDetector
from researchpilot.evidence.contradiction_models import (
    ContradictionAssessment,
)
from researchpilot.evidence.models import Evidence


class ContradictionEvaluator:
    """Evaluate contradictions across research evidence."""

    def __init__(self) -> None:
        self._detector = ContradictionDetector()

    def evaluate(
        self,
        evidence_items: list[Evidence],
    ) -> ContradictionAssessment:
        """Generate a contradiction assessment."""

        contradictions = self._detector.detect(evidence_items)

        return ContradictionAssessment(
            evidence_count=len(evidence_items),
            contradiction_count=len(contradictions),
            contradictions=contradictions,
        )
