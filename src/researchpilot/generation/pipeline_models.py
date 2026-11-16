
"""Models for the research report pipeline."""

from dataclasses import dataclass

from researchpilot.evidence.coverage_models import (
    EvidenceCoverageAssessment,
)
from researchpilot.evidence.contradiction_models import (
    ContradictionAssessment,
)
from researchpilot.evidence.models import EvaluatedEvidence
from researchpilot.evidence.source_models import EvaluatedSource
from researchpilot.evidence.synthesis_models import ResearchSynthesis

from .citation_models import CitationValidationResult
from .report_models import ResearchReport


@dataclass
class ResearchPipelineResult:
    """Aggregated output from the research report pipeline."""

    evaluated_evidence: list[EvaluatedEvidence]
    evaluated_sources: list[EvaluatedSource]
    coverage: EvidenceCoverageAssessment
    contradictions: ContradictionAssessment
    synthesis: ResearchSynthesis
    report: ResearchReport
    citation_validation: CitationValidationResult
