
from pathlib import Path

import httpx
import pytest

from researchpilot.retrieval.fulltext.downloader import (
    PDFDownloadError,
    PDFDownloader,
)

VALID_PDF = b"%PDF-1.4\nmock pdf content"


@pytest.fixture
def downloader(tmp_path: Path) -> PDFDownloader:
    return PDFDownloader(
        storage_dir=tmp_path,
        timeout=5,
        max_size_bytes=1024,
    )


def mock_transport(
    content: bytes,
    status_code: int = 200,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            content=content,
            request=request,
        )

    return httpx.MockTransport(handler)


def patch_httpx_client(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
) -> None:
    """Patch the downloader's HTTP client with a mock transport."""
    original_client = httpx.Client

    monkeypatch.setattr(
        "researchpilot.retrieval.fulltext.downloader.httpx.Client",
        lambda **kwargs: original_client(
            transport=transport,
            **kwargs,
        ),
    )


def test_download_valid_pdf(
    downloader: PDFDownloader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = mock_transport(VALID_PDF)
    patch_httpx_client(monkeypatch, transport)

    path = downloader.download(
        "W123",
        "https://example.org/paper.pdf",
    )

    assert path.exists()
    assert path.read_bytes() == VALID_PDF


def test_rejects_invalid_paper_id(
    downloader: PDFDownloader,
) -> None:
    with pytest.raises(PDFDownloadError, match="Invalid paper ID"):
        downloader.download(
            "../unsafe",
        "https://example.org/paper.pdf",
    )


def test_rejects_non_http_url(
    downloader: PDFDownloader,
) -> None:
    with pytest.raises(PDFDownloadError, match="HTTP and HTTPS"):
        downloader.download(
            "W123",
            "file:///paper.pdf",
        )


def test_rejects_non_pdf_response(
    downloader: PDFDownloader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = mock_transport(b"<html>Not a PDF</html>")
    patch_httpx_client(monkeypatch, transport)

    with pytest.raises(
        PDFDownloadError,
        match="not a valid PDF",
    ):
        downloader.download(
            "W123",
            "https://example.org/paper.pdf",
        )


def test_rejects_pdf_exceeding_max_size(
    downloader: PDFDownloader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = mock_transport(
        VALID_PDF + b"x" * 2048,
    )
    patch_httpx_client(monkeypatch, transport)

    with pytest.raises(
        PDFDownloadError,
        match="maximum size",
    ):
        downloader.download(
            "W123",
            "https://example.org/paper.pdf",
        )


def test_uses_cached_pdf(
    downloader: PDFDownloader,
) -> None:
    cached_pdf = downloader.storage_dir / "W123.pdf"

    downloader.storage_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    cached_pdf.write_bytes(VALID_PDF)

    path = downloader.download(
        "W123",
        "https://example.org/paper.pdf",
    )

    assert path == cached_pdf
    assert path.read_bytes() == VALID_PDF
