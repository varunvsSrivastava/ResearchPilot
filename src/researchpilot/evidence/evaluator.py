
"""Evidence evaluation service."""

from researchpilot.evidence.models import (
    Evidence,
    EvaluatedEvidence,
)
from researchpilot.evidence.quality import EvidenceQualityEvaluator
from researchpilot.rag.vectorstore.models import VectorSearchResult


class EvidenceEvaluator:
    """Convert retrieved results into evaluated research evidence."""

    def __init__(
        self,
        quality_evaluator: EvidenceQualityEvaluator | None = None,
    ) -> None:
        """Initialize evidence evaluation dependencies."""
        self.quality_evaluator = (
            quality_evaluator or EvidenceQualityEvaluator()
        )

    def evaluate(
        self,
        results: list[VectorSearchResult],
        research_question: str,
    ) -> list[EvaluatedEvidence]:
        """Evaluate retrieved results and return structured evidence."""
        evaluated_evidence = []

        for result in results:
            record = result.record

            evidence = Evidence(
                content=record.text,
                source_id=record.id,
                metadata=record.metadata,
                retrieval_score=result.score,
            )

            quality = self.quality_evaluator.evaluate(
                content=evidence.content,
                research_question=research_question,
            )

            evaluated_evidence.append(
                EvaluatedEvidence(
                    evidence=evidence,
                    quality=quality,
                )
            )

        return evaluated_evidence
