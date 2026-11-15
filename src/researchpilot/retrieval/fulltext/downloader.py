
"""Safe streaming downloader for academic PDF files."""

import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

from researchpilot.config import (
    get_pdf_download_timeout,
    get_pdf_max_size_bytes,
    get_pdf_storage_dir,
)


PDF_SIGNATURE = b"%PDF-"
PAPER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class PDFDownloadError(Exception):
    """Raised when a PDF cannot be safely downloaded."""


class PDFDownloader:
    """Download and cache academic PDFs safely."""

    def __init__(
        self,
        storage_dir: Path | None = None,
        timeout: float | None = None,
        max_size_bytes: int | None = None,
    ) -> None:
        self.storage_dir = storage_dir or get_pdf_storage_dir()
        self.timeout = timeout or get_pdf_download_timeout()
        self.max_size_bytes = max_size_bytes or get_pdf_max_size_bytes()

    def download(self, paper_id: str, url: str) -> Path:
        """Download a PDF and return its local path."""
        self._validate_paper_id(paper_id)
        self._validate_url(url)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        destination = self.storage_dir / f"{paper_id}.pdf"

        if self._is_valid_pdf(destination):
            return destination

        try:
            return self._download_to_file(url, destination)
        except httpx.HTTPError as exc:
            raise PDFDownloadError(
                f"PDF download failed for paper {paper_id}: {exc}"
            ) from exc

    def _download_to_file(self, url: str, destination: Path) -> Path:
        temporary_path: Path | None = None

        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()

                    content_length = response.headers.get("content-length")
                    if content_length:
                        if int(content_length) > self.max_size_bytes:
                            raise PDFDownloadError(
                                "PDF exceeds the configured maximum size."
                            )

                    with tempfile.NamedTemporaryFile(
                        dir=self.storage_dir,
                        suffix=".tmp",
                        delete=False,
                    ) as temporary_file:
                        temporary_path = Path(temporary_file.name)
                        total_bytes = 0
                        signature = bytearray()

                        for chunk in response.iter_bytes():
                            if not chunk:
                                continue

                            total_bytes += len(chunk)

                            if total_bytes > self.max_size_bytes:
                                raise PDFDownloadError(
                                    "PDF exceeds the configured maximum size."
                                )

                            if len(signature) < len(PDF_SIGNATURE):
                                signature.extend(
                                    chunk[: len(PDF_SIGNATURE) - len(signature)]
                                )

                            temporary_file.write(chunk)

                        if bytes(signature) != PDF_SIGNATURE:
                            raise PDFDownloadError(
                                "Downloaded content is not a valid PDF."
                            )

                        if total_bytes == 0:
                            raise PDFDownloadError("Downloaded PDF is empty.")

            temporary_path.replace(destination)
            return destination

        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _validate_paper_id(paper_id: str) -> None:
        if not PAPER_ID_PATTERN.fullmatch(paper_id):
            raise PDFDownloadError("Invalid paper ID.")

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed_url = urlparse(url)

        if parsed_url.scheme not in {"http", "https"}:
            raise PDFDownloadError("Only HTTP and HTTPS URLs are supported.")

        if not parsed_url.netloc:
            raise PDFDownloadError("Invalid PDF URL.")

    @staticmethod
    def _is_valid_pdf(path: Path) -> bool:
        if not path.is_file():
            return False

        try:
            with path.open("rb") as pdf_file:
                return pdf_file.read(len(PDF_SIGNATURE)) == PDF_SIGNATURE
        except OSError:
            return False
