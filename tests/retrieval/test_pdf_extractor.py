
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

def test_normalizes_line_breaks_within_paragraph(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "line_breaks.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (72, 72, 400, 150),
            "Retrieval augmented generation improves "
            "factual accuracy in large language models.",
            fontsize=12,
        )
        pdf.save(pdf_path)

    document = PDFDocument(
        paper_id="W124",
        title="Line Break Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=pdf_path,
        file_size_bytes=pdf_path.stat().st_size,
    )

    result = PDFExtractor().extract(document)

    text = result.pages[0].text

    assert "Retrieval augmented generation" in text
    assert "factual accuracy" in text


def test_preserves_paragraph_separation(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paragraphs.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "First research paragraph.")
        page.insert_text((72, 120), "Second research paragraph.")
        pdf.save(pdf_path)

    document = PDFDocument(
        paper_id="W125",
        title="Paragraph Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=pdf_path,
        file_size_bytes=pdf_path.stat().st_size,
    )

    result = PDFExtractor().extract(document)

    text = result.pages[0].text

    assert "First research paragraph." in text
    assert "Second research paragraph." in text

def test_preserves_separation_between_text_blocks(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "separated_blocks.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "First text block.")
        page.insert_text((72, 120), "Second text block.")
        pdf.save(pdf_path)

    document = PDFDocument(
        paper_id="W126",
        title="Block Separation Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=pdf_path,
        file_size_bytes=pdf_path.stat().st_size,
    )

    result = PDFExtractor().extract(document)

    assert "First text block.\n\nSecond text block." in (
        result.pages[0].text
    )


def test_excludes_empty_text_blocks(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "empty_blocks.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Research content.")
        pdf.save(pdf_path)

    document = PDFDocument(
        paper_id="W127",
        title="Empty Block Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=pdf_path,
        file_size_bytes=pdf_path.stat().st_size,
    )

    result = PDFExtractor().extract(document)

    assert result.pages[0].text == "Research content."
