
"""Research synthesis service."""

from researchpilot.evidence.models import EvaluatedEvidence
from researchpilot.evidence.synthesis import ResearchSynthesizer
from researchpilot.evidence.synthesis_models import ResearchSynthesis


class ResearchSynthesisService:
    """Generate structured research synthesis."""

    def __init__(self) -> None:
        self._synthesizer = ResearchSynthesizer()

    def synthesize(
        self,
        research_question: str,
        evaluated_evidence: list[EvaluatedEvidence],
    ) -> ResearchSynthesis:
        """Generate a structured synthesis from evaluated evidence."""

        if not research_question.strip():
            raise ValueError("Research question must not be empty.")

        findings = self._synthesizer.synthesize_findings(
            evaluated_evidence
        )

        summary = (
            f"Generated {len(findings)} findings from "
            f"{len(evaluated_evidence)} evidence items."
        )

        return ResearchSynthesis(
            research_question=research_question,
            findings=findings,
            evidence_count=len(evaluated_evidence),
            synthesis_summary=summary,
        )
