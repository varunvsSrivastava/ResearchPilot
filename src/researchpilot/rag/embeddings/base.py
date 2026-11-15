
"""Abstract interface for embedding providers."""

from abc import ABC, abstractmethod

from researchpilot.rag.embeddings.models import (
    EmbeddingBatch,
    EmbeddingVector,
)


class EmbeddingProvider(ABC):
    """Contract for text embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingVector:
        """Generate an embedding for a single text."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        """Generate embeddings for multiple texts."""
