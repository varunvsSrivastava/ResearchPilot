
"""Research evidence synthesis utilities."""

from researchpilot.evidence.models import EvaluatedEvidence
from researchpilot.evidence.synthesis_models import (
    ResearchFinding,
)


class ResearchSynthesizer:
    """Synthesize evaluated evidence into research findings."""

    def synthesize_findings(
        self,
        evaluated_evidence: list[EvaluatedEvidence],
    ) -> list[ResearchFinding]:
        """Aggregate evaluated evidence into individual findings."""

        findings = []

        for item in evaluated_evidence:
            evidence = item.evidence
            quality = item.quality

            findings.append(
                ResearchFinding(
                    finding=evidence.content,
                    supporting_sources=[evidence.source_id],
                    evidence_count=1,
                    confidence_score=quality.overall_score,
                )
            )

        return findings
