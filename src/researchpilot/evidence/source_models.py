"""Models for academic source credibility evaluation."""

from pydantic import BaseModel, Field


class SourceCredibilityScore(BaseModel):
    """Represent academic source credibility across multiple dimensions."""

    doi_score: float = Field(ge=0.0, le=1.0)
    citation_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(ge=0.0, le=1.0)
    metadata_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)


class EvaluatedSource(BaseModel):
    """Represent an academic paper with its credibility assessment."""

    paper_id: str
    title: str
    provider: str
    credibility: SourceCredibilityScore