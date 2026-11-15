
"""Data models for embedding operations."""

from pydantic import BaseModel, Field


class EmbeddingVector(BaseModel):
    """Vector representation of a text input."""

    text: str

    vector: list[float] = Field(
        min_length=1,
        description="Dense numerical embedding vector.",
    )

    dimensions: int = Field(
        ge=1,
        description="Number of dimensions in the embedding vector.",
    )

    model: str = Field(
        min_length=1,
        description="Embedding model used to generate the vector.",
    )

    @property
    def is_valid(self) -> bool:
        """Check whether vector dimensions match metadata."""
        return len(self.vector) == self.dimensions


class EmbeddingBatch(BaseModel):
    """Collection of embedding vectors."""

    embeddings: list[EmbeddingVector] = Field(
        default_factory=list,
    )

    model: str

    total_embeddings: int = 0

    @property
    def is_valid(self) -> bool:
        """Check whether batch metadata is consistent."""
        return (
            self.total_embeddings == len(self.embeddings)
            and all(embedding.is_valid for embedding in self.embeddings)
        )
