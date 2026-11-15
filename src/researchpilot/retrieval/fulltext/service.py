
"""Orchestration service for academic full-text acquisition."""

from researchpilot.retrieval.models import Paper
from researchpilot.retrieval.fulltext.chunker import TextChunker
from researchpilot.retrieval.fulltext.downloader import (
    PDFDownloader,
    PDFDownloadError,
)
from researchpilot.retrieval.fulltext.evidence import EvidenceGenerator
from researchpilot.retrieval.fulltext.extractor import PDFExtractor
from researchpilot.retrieval.fulltext.models import Evidence, PDFDocument
from researchpilot.retrieval.fulltext.normalizer import TextNormalizer


class FullTextAcquisitionError(Exception):
    """Raised when full-text acquisition cannot be completed."""


class FullTextAcquisitionService:
    """Acquire academic PDFs and convert them into evidence records."""

    def __init__(
        self,
        downloader: PDFDownloader | None = None,
        extractor: PDFExtractor | None = None,
        normalizer: TextNormalizer | None = None,
        chunker: TextChunker | None = None,
        evidence_generator: EvidenceGenerator | None = None,
    ) -> None:
        self.downloader = downloader or PDFDownloader()
        self.extractor = extractor or PDFExtractor()
        self.normalizer = normalizer or TextNormalizer()
        self.chunker = chunker or TextChunker()
        self.evidence_generator = (
            evidence_generator or EvidenceGenerator()
        )

    def acquire(self, paper: Paper) -> list[Evidence]:
        """Acquire a paper's full text and generate evidence records."""
        pdf_urls = self._get_pdf_urls(paper)

        if not pdf_urls:
            raise FullTextAcquisitionError(
                f"No PDF URLs available for paper {paper.paper_id}."
            )

        download_errors: list[str] = []

        for pdf_url in pdf_urls:
            try:
                file_path = self.downloader.download(
                    paper.paper_id,
                    pdf_url,
                )
            except PDFDownloadError as exc:
                download_errors.append(str(exc))
                continue

            pdf_document = PDFDocument(
                paper_id=paper.paper_id,
                title=paper.title,
                doi=paper.doi,
                source_url=paper.source_url,
                pdf_url=pdf_url,
                file_path=file_path,
                file_size_bytes=file_path.stat().st_size,
            )

            extracted_document = self.extractor.extract(pdf_document)
            normalized_document = self.normalizer.normalize_document(
                extracted_document
            )
            chunks = self.chunker.chunk_document(normalized_document)

            return self.evidence_generator.generate_many(chunks)

        raise FullTextAcquisitionError(
            f"Failed to download PDF for paper {paper.paper_id}. "
            f"Attempts: {'; '.join(download_errors)}"
        )

    @staticmethod
    def _get_pdf_urls(paper: Paper) -> list[str]:
        """Return unique PDF URLs while preserving their priority."""
        urls = [*paper.pdf_urls]

        if paper.pdf_url:
            urls.append(paper.pdf_url)

        return list(dict.fromkeys(url for url in urls if url))
