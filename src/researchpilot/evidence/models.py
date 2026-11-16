
"""Models for research evidence and quality evaluation."""

from pydantic import BaseModel, Field

from researchpilot.rag.vectorstore.models import VectorSearchResult


class Evidence(BaseModel):
    """Represent a retrieved item as research evidence."""

    content: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict
    )
    retrieval_score: float


class EvidenceQualityScore(BaseModel):
    """Represent evidence quality across independent dimensions."""

    relevance: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)


class EvaluatedEvidence(BaseModel):
    """Represent evidence together with its quality assessment."""

    evidence: Evidence
    quality: EvidenceQualityScore
