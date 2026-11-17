
"""API request schemas for research execution."""

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request payload for executing a research workflow."""

    research_question: str = Field(
        min_length=3,
        max_length=2000,
        description="The research question to investigate.",
    )
