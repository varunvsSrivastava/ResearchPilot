
"""Abstract interface for semantic retrieval."""

from abc import ABC, abstractmethod

from researchpilot.rag.retrieval.models import (
    RetrievalRequest,
    RetrievalResponse,
)


class RetrievalService(ABC):
    """Contract for semantic retrieval services."""

    @abstractmethod
    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        """Retrieve relevant evidence for a research query."""
