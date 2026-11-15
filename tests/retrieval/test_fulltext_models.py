
from pathlib import Path

import pytest
from pydantic import ValidationError

from researchpilot.retrieval.fulltext.models import PDFDocument


def test_pdf_document_creation():
    document = PDFDocument(
        paper_id="W123",
        title="Research Paper",
        doi="10.1234/example",
        source_url="https://openalex.org/W123",
        pdf_url="https://example.com/paper.pdf",
        file_path=Path("data/papers/W123.pdf"),
        file_size_bytes=1024,
    )

    assert document.paper_id == "W123"
    assert document.title == "Research Paper"
    assert document.file_path == Path("data/papers/W123.pdf")
    assert document.file_size_bytes == 1024


def test_pdf_document_allows_missing_doi():
    document = PDFDocument(
        paper_id="W123",
        title="Research Paper",
        source_url="https://openalex.org/W123",
        pdf_url="https://example.com/paper.pdf",
        file_path=Path("data/papers/W123.pdf"),
        file_size_bytes=1024,
    )

    assert document.doi is None


def test_pdf_document_rejects_non_positive_file_size():
    with pytest.raises(ValidationError):
        PDFDocument(
            paper_id="W123",
            title="Research Paper",
            source_url="https://openalex.org/W123",
            pdf_url="https://example.com/paper.pdf",
            file_path=Path("data/papers/W123.pdf"),
            file_size_bytes=0,
        )
