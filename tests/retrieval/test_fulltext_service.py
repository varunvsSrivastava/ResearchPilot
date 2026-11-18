
from pathlib import Path
from unittest.mock import Mock

import pytest

from researchpilot.retrieval.fulltext.downloader import PDFDownloadError
from researchpilot.retrieval.fulltext.models import Evidence
from researchpilot.retrieval.fulltext.service import (
    FullTextAcquisitionError,
    FullTextAcquisitionService,
)
from researchpilot.retrieval.models import Paper
from researchpilot.retrieval.fulltext.chunker import TextChunker
from researchpilot.retrieval.fulltext.evidence import EvidenceGenerator
from researchpilot.retrieval.fulltext.models import (
    ExtractedDocument,
    ExtractedPage,
)
from researchpilot.retrieval.fulltext.normalizer import TextNormalizer

@pytest.fixture
def paper() -> Paper:
    return Paper(
        paper_id="W123",
        provider="openalex",
        title="Sample Research Paper",
        doi="10.1234/example",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        pdf_urls=[
            "https://example.org/paper.pdf",
            "https://example.org/backup.pdf",
        ],
        search_query="AI research",
    )


@pytest.fixture
def evidence() -> Evidence:
    return Evidence(
        evidence_id="ev_test123",
        paper_id="W123",
        title="Sample Research Paper",
        doi="10.1234/example",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        page_number=1,
        chunk_index=0,
        text="Sample evidence text.",
        start_char=0,
        end_char=21,
    )


@pytest.fixture
def service() -> FullTextAcquisitionService:
    return FullTextAcquisitionService(
        downloader=Mock(),
        extractor=Mock(),
        normalizer=Mock(),
        chunker=Mock(),
        evidence_generator=Mock(),
    )


def test_acquire_success(
    service: FullTextAcquisitionService,
    paper: Paper,
    evidence: Evidence,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-sample")

    extracted_document = Mock()
    normalized_document = Mock()
    chunks = [Mock()]

    service.downloader.download.return_value = pdf_path
    service.extractor.extract.return_value = extracted_document
    service.normalizer.normalize_document.return_value = normalized_document
    service.chunker.chunk_document.return_value = chunks
    service.evidence_generator.generate_many.return_value = [evidence]

    result = service.acquire(paper)

    assert result == [evidence]
    service.downloader.download.assert_called_once_with(
        paper.paper_id,
        paper.pdf_urls[0],
    )
    service.extractor.extract.assert_called_once()
    service.normalizer.normalize_document.assert_called_once_with(
        extracted_document
    )
    service.chunker.chunk_document.assert_called_once_with(
        normalized_document
    )
    service.evidence_generator.generate_many.assert_called_once_with(
        chunks
    )


def test_acquire_falls_back_to_next_pdf_url(
    service: FullTextAcquisitionService,
    paper: Paper,
    evidence: Evidence,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-sample")

    service.downloader.download.side_effect = [
        PDFDownloadError("Primary URL failed"),
        pdf_path,
    ]
    service.extractor.extract.return_value = Mock()
    service.normalizer.normalize_document.return_value = Mock()
    service.chunker.chunk_document.return_value = [Mock()]
    service.evidence_generator.generate_many.return_value = [evidence]

    result = service.acquire(paper)

    assert result == [evidence]
    assert service.downloader.download.call_count == 2
    service.downloader.download.assert_any_call(
        paper.paper_id,
        paper.pdf_urls[0],
    )
    service.downloader.download.assert_any_call(
        paper.paper_id,
        paper.pdf_urls[1],
    )


def test_acquire_rejects_paper_without_pdf_urls(
    service: FullTextAcquisitionService,
    paper: Paper,
) -> None:
    paper = paper.model_copy(
        update={"pdf_url": None, "pdf_urls": []}
    )

    with pytest.raises(FullTextAcquisitionError, match="No PDF URLs"):
        service.acquire(paper)

    service.downloader.download.assert_not_called()


def test_acquire_fails_when_all_downloads_fail(
    service: FullTextAcquisitionService,
    paper: Paper,
) -> None:
    service.downloader.download.side_effect = PDFDownloadError(
        "Download failed"
    )

    with pytest.raises(
        FullTextAcquisitionError,
        match="Failed to download PDF",
    ):
        service.acquire(paper)

    assert service.downloader.download.call_count == 2


def test_acquire_returns_empty_when_no_chunks(
    service: FullTextAcquisitionService,
    paper: Paper,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-sample")

    service.downloader.download.return_value = pdf_path
    service.extractor.extract.return_value = Mock()
    service.normalizer.normalize_document.return_value = Mock()
    service.chunker.chunk_document.return_value = []
    service.evidence_generator.generate_many.return_value = []

    result = service.acquire(paper)

    assert result == []
    service.evidence_generator.generate_many.assert_called_once_with([])

def test_acquire_preserves_normalized_evidence(
    paper: Paper,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-sample")

    extracted_document = ExtractedDocument(
        paper_id=paper.paper_id,
        title=paper.title,
        doi=paper.doi,
        source_url=paper.source_url,
        pdf_url=paper.pdf_url,
        file_path=pdf_path,
        pages=[
            ExtractedPage(
                page_number=1,
                text=(
                    "Retrieval aug-\n"
                    "mented generation improves factual accuracy.\n\n"
                    "It supports evidence-based research."
                ),
            )
        ],
    )

    extractor = Mock()
    extractor.extract.return_value = extracted_document

    service = FullTextAcquisitionService(
        downloader=Mock(),
        extractor=extractor,
        normalizer=TextNormalizer(),
        chunker=TextChunker(
            chunk_size=100,
            overlap=20,
        ),
        evidence_generator=EvidenceGenerator(),
    )

    service.downloader.download.return_value = pdf_path

    evidence_records = service.acquire(paper)

    assert len(evidence_records) >= 1

    combined_text = " ".join(
        evidence.text for evidence in evidence_records
    )

    assert "augmented generation" in combined_text
    assert "factual accuracy" in combined_text
    assert "evidence-based research" in combined_text

    assert all(
        evidence.paper_id == paper.paper_id
        for evidence in evidence_records
    )

    assert all(
        evidence.page_number == 1
        for evidence in evidence_records
    )

    assert all(
        evidence.start_char < evidence.end_char
        for evidence in evidence_records
    )