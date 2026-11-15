
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
