
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Normalized metadata for an academic paper."""

    paper_id: str
    provider: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    doi: str | None = None
    source_url: str
    pdf_url: str | None = None
    citation_count: int | None = None
    is_open_access: bool | None = None
    search_query: str


class SearchResult(BaseModel):
    """Results returned by an academic search operation."""

    query: str
    provider: str
    papers: list[Paper] = Field(default_factory=list)
    total_results: int | None = None
