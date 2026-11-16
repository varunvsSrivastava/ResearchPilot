
"""Models for evidence contradiction detection."""

from pydantic import BaseModel, Field


class EvidenceContradiction(BaseModel):
    """Represent a contradiction between two evidence items."""

    first_source_id: str = Field(min_length=1)
    second_source_id: str = Field(min_length=1)
    first_claim: str = Field(min_length=1)
    second_claim: str = Field(min_length=1)
    contradiction_score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)


class ContradictionAssessment(BaseModel):
    """Represent contradiction detection results."""

    evidence_count: int = Field(ge=0)
    contradiction_count: int = Field(ge=0)
    contradictions: list[EvidenceContradiction] = Field(
        default_factory=list
    )
