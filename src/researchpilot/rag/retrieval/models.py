
"""Models for semantic retrieval."""

from pydantic import BaseModel, Field

from researchpilot.rag.vectorstore.models import VectorSearchResult


class RetrievalRequest(BaseModel):
    """Represent a semantic retrieval request."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)


class RetrievalResponse(BaseModel):
    """Represent the outcome of semantic retrieval."""

    query: str
    results: list[VectorSearchResult] = Field(default_factory=list)
    total_results: int = 0

    @property
    def is_valid(self) -> bool:
        """Validate result count consistency."""
        return self.total_results == len(self.results)
