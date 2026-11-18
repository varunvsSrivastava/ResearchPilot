
"""PDF text extraction using PyMuPDF."""

from pathlib import Path

import pymupdf

from researchpilot.retrieval.fulltext.models import (
    ExtractedDocument,
    ExtractedPage,
    PDFDocument,
)

class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""

class PDFExtractor:
    """Extract page-wise text blocks from academic PDF documents."""

    def extract(self, document: PDFDocument) -> ExtractedDocument:
        """Extract text blocks from every PDF page."""
        pdf_path = Path(document.file_path)

        if not pdf_path.is_file():
            raise PDFExtractionError(
                f"PDF file does not exist: {pdf_path}"
            )

        try:
            with pymupdf.open(pdf_path) as pdf:
                if not pdf.is_pdf:
                    raise PDFExtractionError(
                        f"File is not a PDF: {pdf_path}"
                    )

                pages = []

                for page_number, page in enumerate(pdf, start=1):
                    blocks = page.get_text("blocks")

                    page_text = "\n\n".join(
                        block[4].strip()
                        for block in blocks
                        if block[4].strip()
                    )

                    pages.append(
                        ExtractedPage(
                            page_number=page_number,
                            text=page_text,
                        )
                    )

        except PDFExtractionError:
            raise

        except (pymupdf.FileDataError, RuntimeError, OSError) as exc:
            raise PDFExtractionError(
                f"Failed to extract text from PDF: {pdf_path}"
            ) from exc

        return ExtractedDocument(
            paper_id=document.paper_id,
            title=document.title,
            doi=document.doi,
            source_url=document.source_url,
            pdf_url=document.pdf_url,
            file_path=pdf_path,
            pages=pages,
        )
