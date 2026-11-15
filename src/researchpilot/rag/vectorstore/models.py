
"""Models for vector store operations."""

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """Represent a document record stored with its embedding."""

    id: str = Field(min_length=1)
    text: str
    vector: list[float] = Field(min_length=1)
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict
    )

    @property
    def dimensions(self) -> int:
        """Return the vector dimensions."""
        return len(self.vector)


class VectorSearchResult(BaseModel):
    """Represent a vector similarity search result."""

    record: VectorRecord
    score: float


class VectorSearchResponse(BaseModel):
    """Represent a vector search response."""

    results: list[VectorSearchResult] = Field(default_factory=list)
    total_results: int = 0

    @property
    def is_valid(self) -> bool:
        """Validate result count consistency."""
        return self.total_results == len(self.results)
