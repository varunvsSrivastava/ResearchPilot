
"""Research execution API routes."""

from fastapi import APIRouter, Depends

from researchpilot.agent.orchestrator import ResearchOrchestrator
from researchpilot.api.dependencies import get_research_orchestrator
from researchpilot.api.schemas.request import ResearchRequest
from researchpilot.api.schemas.response import (
    ResearchPipelineResponse,
    ResearchResponse,
)


router = APIRouter(prefix="/api/v1/research", tags=["Research"])


@router.post("", response_model=ResearchResponse)
def execute_research(
    request: ResearchRequest,
    orchestrator: ResearchOrchestrator = Depends(
        get_research_orchestrator
    ),
) -> ResearchResponse:
    """Execute the complete research workflow."""

    agent = orchestrator.execute_full_research(
        request.research_question
    )

    research_result = None

    if agent.research_result is not None:
        research_result = ResearchPipelineResponse.model_validate(
            agent.research_result,
            from_attributes=True,
        )

    return ResearchResponse(
        research_question=agent.state.research_question,
        status=agent.state.status,
        research_plan=agent.state.research_plan,
        steps=agent.state.steps,
        final_answer=agent.state.final_answer,
        research_result=research_result,
        error=agent.state.error,
    )
