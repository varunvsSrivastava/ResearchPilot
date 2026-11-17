
"""Models for evidence indexing."""

from pydantic import BaseModel, Field


class IndexingResult(BaseModel):
    """Represent the outcome of an evidence indexing operation."""

    total_evidence: int = 0
    indexed_evidence: int = 0
    skipped_evidence: int = 0
    failed_evidence: int = 0
    errors: list[str] = Field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        """Return whether all evidence was indexed or already available."""
        return (
            self.failed_evidence == 0
            and (
                self.indexed_evidence + self.skipped_evidence
                == self.total_evidence
            )
        )


class IndexingRequest(BaseModel):
    """Represent an evidence indexing request."""

    evidence_ids: list[str] = Field(default_factory=list)
    batch_size: int = Field(default=100, ge=1)
    overwrite: bool = False
