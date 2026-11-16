
"""Models for evidence coverage analysis."""

from pydantic import BaseModel, Field


class EvidenceCoverageScore(BaseModel):
    """Represent evidence coverage for a research question."""

    coverage_score: float = Field(ge=0.0, le=1.0)
    covered_aspects: list[str] = Field(default_factory=list)
    uncovered_aspects: list[str] = Field(default_factory=list)
    total_aspects: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0.0, le=1.0)


class EvidenceCoverageAssessment(BaseModel):
    """Represent evidence coverage assessment results."""

    research_question: str = Field(min_length=1)
    evidence_count: int = Field(ge=0)
    coverage: EvidenceCoverageScore
