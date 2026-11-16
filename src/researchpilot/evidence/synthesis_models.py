
"""Models for research evidence synthesis."""

from pydantic import BaseModel, Field


class ResearchFinding(BaseModel):
    """Represent a synthesized research finding."""

    finding: str = Field(min_length=1)
    supporting_sources: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    confidence_score: float = Field(ge=0.0, le=1.0)


class ResearchSynthesis(BaseModel):
    """Represent a structured synthesis of research evidence."""

    research_question: str = Field(min_length=1)
    findings: list[ResearchFinding] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    synthesis_summary: str = Field(min_length=1)
