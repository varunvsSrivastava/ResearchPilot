
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    return api_key


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_pdf_storage_dir() -> Path:
    """Return the project-relative PDF storage directory."""
    storage_dir = Path(
        os.getenv("PDF_STORAGE_DIR", "data/papers")
    )

    if not storage_dir.is_absolute():
        storage_dir = PROJECT_ROOT / storage_dir

    return storage_dir.resolve()


def get_pdf_download_timeout() -> float:
    """Return the PDF download timeout in seconds."""
    return float(os.getenv("PDF_DOWNLOAD_TIMEOUT", "30"))


def get_pdf_max_size_bytes() -> int:
    """Return the maximum allowed PDF size in bytes."""
    return int(os.getenv("PDF_MAX_SIZE_BYTES", str(50 * 1024 * 1024)))
