
"""Models for academic full-text documents."""

from pathlib import Path

from pydantic import BaseModel, Field


class PDFDocument(BaseModel):
    """Metadata for an acquired academic PDF document."""

    paper_id: str
    title: str
    doi: str | None = None
    source_url: str
    pdf_url: str
    file_path: Path
    file_size_bytes: int = Field(gt=0)


class ExtractedPage(BaseModel):
    """Text extracted from a single PDF page."""

    page_number: int = Field(ge=1)
    text: str


class ExtractedDocument(BaseModel):
    """Text extracted from an academic PDF document."""

    paper_id: str
    title: str
    doi: str | None = None
    source_url: str
    pdf_url: str
    file_path: Path
    pages: list[ExtractedPage]

    @property
    def page_count(self) -> int:
        """Return the number of extracted pages."""
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """Combine page text while preserving page boundaries."""
        return "\n\n".join(page.text for page in self.pages)
