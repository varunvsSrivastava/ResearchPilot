
"""API response schemas for research execution."""

from pydantic import BaseModel, ConfigDict

from researchpilot.agent.models import AgentStatus, AgentStep
from researchpilot.schemas.research_plan import ResearchPlan

from researchpilot.evidence.models import EvaluatedEvidence
from researchpilot.evidence.source_models import EvaluatedSource
from researchpilot.evidence.coverage_models import EvidenceCoverageAssessment
from researchpilot.evidence.contradiction_models import (
    ContradictionAssessment,
)
from researchpilot.evidence.synthesis_models import ResearchSynthesis

from researchpilot.generation.report_models import ResearchReport
from researchpilot.generation.citation_models import (
    CitationValidationResult,
)


class ResearchPipelineResponse(BaseModel):
    """Structured output from the research report pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    evaluated_evidence: list[EvaluatedEvidence]
    evaluated_sources: list[EvaluatedSource]
    coverage: EvidenceCoverageAssessment
    contradictions: ContradictionAssessment
    synthesis: ResearchSynthesis
    report: ResearchReport
    citation_validation: CitationValidationResult


class ResearchResponse(BaseModel):
    """Complete response returned by the research API."""

    research_question: str
    status: AgentStatus
    research_plan: ResearchPlan | None = None
    steps: list[AgentStep]
    final_answer: str | None = None
    research_result: ResearchPipelineResponse | None = None
    error: str | None = None
