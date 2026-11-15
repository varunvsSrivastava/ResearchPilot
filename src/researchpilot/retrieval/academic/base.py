
from abc import ABC, abstractmethod

from researchpilot.retrieval.models import Paper


class AcademicSearchProvider(ABC):
    """Abstract interface for academic paper search providers."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Search academic papers and return normalized results."""
        raise NotImplementedError
