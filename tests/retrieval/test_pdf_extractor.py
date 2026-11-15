
from pathlib import Path

import pymupdf
import pytest

from researchpilot.retrieval.fulltext.extractor import (
    PDFExtractionError,
    PDFExtractor,
)
from researchpilot.retrieval.fulltext.models import PDFDocument


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a two-page PDF containing extractable text."""
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page1 = pdf.new_page()
        page1.insert_text((72, 72), "ResearchPilot page one")

        page2 = pdf.new_page()
        page2.insert_text((72, 72), "ResearchPilot page two")

        pdf.save(pdf_path)

    return pdf_path


@pytest.fixture
def pdf_document(sample_pdf: Path) -> PDFDocument:
    return PDFDocument(
        paper_id="W123",
        title="Sample Research Paper",
        doi="10.1234/example",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=sample_pdf,
        file_size_bytes=sample_pdf.stat().st_size,
    )


def test_extracts_pdf_pages(
    pdf_document: PDFDocument,
) -> None:
    extractor = PDFExtractor()

    result = extractor.extract(pdf_document)

    assert result.paper_id == "W123"
    assert result.title == "Sample Research Paper"
    assert result.page_count == 2
    assert len(result.pages) == 2


def test_preserves_page_numbers(
    pdf_document: PDFDocument,
) -> None:
    result = PDFExtractor().extract(pdf_document)

    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2


def test_extracts_page_text(
    pdf_document: PDFDocument,
) -> None:
    result = PDFExtractor().extract(pdf_document)

    assert "ResearchPilot page one" in result.pages[0].text
    assert "ResearchPilot page two" in result.pages[1].text


def test_combines_full_text(
    pdf_document: PDFDocument,
) -> None:
    result = PDFExtractor().extract(pdf_document)

    assert "ResearchPilot page one" in result.full_text
    assert "ResearchPilot page two" in result.full_text
    assert "\n\n" in result.full_text


def test_rejects_missing_pdf(
    tmp_path: Path,
) -> None:
    document = PDFDocument(
        paper_id="W123",
        title="Missing Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=tmp_path / "missing.pdf",
        file_size_bytes=100,
    )

    with pytest.raises(
        PDFExtractionError,
        match="does not exist",
    ):
        PDFExtractor().extract(document)


def test_rejects_invalid_pdf(
    tmp_path: Path,
) -> None:
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_bytes(b"This is not a PDF")

    document = PDFDocument(
        paper_id="W123",
        title="Invalid Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=invalid_pdf,
        file_size_bytes=invalid_pdf.stat().st_size,
    )

    with pytest.raises(PDFExtractionError):
        PDFExtractor().extract(document)
