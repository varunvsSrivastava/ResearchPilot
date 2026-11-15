
"""Abstract interface for vector stores."""

from abc import ABC, abstractmethod

from researchpilot.rag.vectorstore.models import (
    VectorRecord,
    VectorSearchResponse,
)


class VectorStore(ABC):
    """Define the contract for vector storage and retrieval."""

    @abstractmethod
    def add_records(self, records: list[VectorRecord]) -> None:
        """Add records to the vector store."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> VectorSearchResponse:
        """Search for similar vectors."""

    @abstractmethod
    def delete_records(self, record_ids: list[str]) -> None:
        """Delete records by their identifiers."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the vector store."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a persisted vector store."""
