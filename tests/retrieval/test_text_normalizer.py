
from pathlib import Path

import pytest

from researchpilot.retrieval.fulltext.models import (
    ExtractedDocument,
    ExtractedPage,
)
from researchpilot.retrieval.fulltext.normalizer import TextNormalizer


@pytest.fixture
def normalizer() -> TextNormalizer:
    return TextNormalizer()


@pytest.fixture
def extracted_document() -> ExtractedDocument:
    return ExtractedDocument(
        paper_id="W123",
        title="Sample Research Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=Path("sample.pdf"),
        pages=[
            ExtractedPage(
                page_number=1,
                text="ResearchPilot   is\tan AI\nresearch assistant.",
            ),
            ExtractedPage(
                page_number=2,
                text="Second page\n\nSecond paragraph.",
            ),
        ],
    )


def test_normalizes_multiple_spaces(
    normalizer: TextNormalizer,
) -> None:
    result = normalizer.normalize_page(
        "ResearchPilot    is     useful"
    )

    assert result == "ResearchPilot is useful"


def test_normalizes_tabs(
    normalizer: TextNormalizer,
) -> None:
    result = normalizer.normalize_page(
        "ResearchPilot\tis\tuseful"
    )

    assert result == "ResearchPilot is useful"


def test_joins_wrapped_lines(
    normalizer: TextNormalizer,
) -> None:
    result = normalizer.normalize_page(
        "ResearchPilot is an\nAI research assistant."
    )

    assert result == "ResearchPilot is an AI research assistant."


def test_preserves_paragraph_boundaries(
    normalizer: TextNormalizer,
) -> None:
    result = normalizer.normalize_page(
        "First paragraph.\n\nSecond paragraph."
    )

    assert result == "First paragraph.\n\nSecond paragraph."


def test_reconstructs_hyphenated_words(
    normalizer: TextNormalizer,
) -> None:
    result = normalizer.normalize_page(
        "This is an inter-\nnational research paper."
    )

    assert result == "This is an international research paper."


def test_normalizes_empty_text(
    normalizer: TextNormalizer,
) -> None:
    assert normalizer.normalize_page("") == ""
    assert normalizer.normalize_page("   \n   ") == ""


def test_normalizes_document_preserving_pages(
    normalizer: TextNormalizer,
    extracted_document: ExtractedDocument,
) -> None:
    result = normalizer.normalize_document(extracted_document)

    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2

    assert result.pages[0].text == (
        "ResearchPilot is an AI research assistant."
    )
    assert result.pages[1].text == (
        "Second page\n\nSecond paragraph."
    )


def test_normalization_does_not_modify_original(
    normalizer: TextNormalizer,
    extracted_document: ExtractedDocument,
) -> None:
    original_text = extracted_document.pages[0].text

    normalizer.normalize_document(extracted_document)

    assert extracted_document.pages[0].text == original_text
