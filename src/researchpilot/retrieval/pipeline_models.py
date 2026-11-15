"""Models for the integrated retrieval pipeline."""

from pydantic import BaseModel, Field

from researchpilot.retrieval.fulltext.models import Evidence
from researchpilot.retrieval.models import Paper


class PaperRetrievalResult(BaseModel):
    """Full-text retrieval result for an individual paper."""

    paper: Paper

    evidence: list[Evidence] = Field(default_factory=list)

    error: str | None = None

    @property
    def success(self) -> bool:
        """Return whether evidence retrieval succeeded."""
        return bool(self.evidence) and self.error is None


class RetrievalPipelineResult(BaseModel):
    """Aggregated output of the retrieval pipeline."""

    research_question: str

    papers: list[Paper] = Field(default_factory=list)

    paper_results: list[PaperRetrievalResult] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(default_factory=list)

    total_papers: int = 0

    successful_papers: int = 0

    failed_papers: int = 0

    total_evidence: int = 0