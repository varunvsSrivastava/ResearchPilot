
"""Orchestrate evidence evaluation and research report generation."""

from researchpilot.evidence.coverage_evaluator import (
    EvidenceCoverageEvaluator,
)
from researchpilot.evidence.contradiction_evaluator import (
    ContradictionEvaluator,
)
from researchpilot.evidence.evaluator import EvidenceEvaluator
from researchpilot.evidence.models import EvaluatedEvidence
from researchpilot.evidence.source_evaluator import SourceEvaluator
from researchpilot.evidence.synthesis_service import ResearchSynthesisService
from researchpilot.rag.vectorstore.models import VectorSearchResult
from researchpilot.retrieval.models import Paper

from .citation_validator import ReportCitationValidator
from .pipeline_models import ResearchPipelineResult
from .report_generator import ReportGenerator


class ResearchReportPipeline:
    """Coordinate evidence evaluation, synthesis, and report generation."""

    def __init__(
        self,
        evidence_evaluator: EvidenceEvaluator | None = None,
        source_evaluator: SourceEvaluator | None = None,
        coverage_evaluator: EvidenceCoverageEvaluator | None = None,
        contradiction_evaluator: ContradictionEvaluator | None = None,
        synthesis_service: ResearchSynthesisService | None = None,
        report_generator: ReportGenerator | None = None,
        citation_validator: ReportCitationValidator | None = None,
    ) -> None:
        """Initialize pipeline dependencies."""

        self.evidence_evaluator = (
            evidence_evaluator or EvidenceEvaluator()
        )
        self.source_evaluator = source_evaluator or SourceEvaluator()
        self.coverage_evaluator = (
            coverage_evaluator or EvidenceCoverageEvaluator()
        )
        self.contradiction_evaluator = (
            contradiction_evaluator or ContradictionEvaluator()
        )
        self.synthesis_service = (
            synthesis_service or ResearchSynthesisService()
        )
        self.report_generator = report_generator or ReportGenerator()
        self.citation_validator = (
            citation_validator or ReportCitationValidator()
        )

    def run(
        self,
        research_question: str,
        retrieved_results: list[VectorSearchResult],
        papers: list[Paper],
        context: str,
        covered_aspects: list[str],
        all_aspects: list[str],
        title: str | None = None,
    ) -> ResearchPipelineResult:
        """Run the complete evidence-to-report workflow."""

        evaluated_evidence = self.evidence_evaluator.evaluate(
            results=retrieved_results,
            research_question=research_question,
        )

        evaluated_sources = self.source_evaluator.evaluate(papers)

        coverage = self.coverage_evaluator.evaluate(
            research_question=research_question,
            evidence_count=len(evaluated_evidence),
            covered_aspects=covered_aspects,
            all_aspects=all_aspects,
        )

        evidence_items = [
            item.evidence for item in evaluated_evidence
        ]

        contradictions = self.contradiction_evaluator.evaluate(
            evidence_items
        )

        synthesis = self.synthesis_service.synthesize(
            research_question=research_question,
            evaluated_evidence=evaluated_evidence,
        )

        report = self.report_generator.generate(
            synthesis=synthesis,
            context=context,
            title=title,
        )

        citation_validation = self.citation_validator.validate(
            report=report,
            context=context,
        )

        return ResearchPipelineResult(
            evaluated_evidence=evaluated_evidence,
            evaluated_sources=evaluated_sources,
            coverage=coverage,
            contradictions=contradictions,
            synthesis=synthesis,
            report=report,
            citation_validation=citation_validation,
        )
