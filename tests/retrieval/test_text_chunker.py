
from pathlib import Path

import pytest

from researchpilot.retrieval.fulltext.chunker import TextChunker
from researchpilot.retrieval.fulltext.models import (
    ExtractedDocument,
    ExtractedPage,
)


@pytest.fixture
def document() -> ExtractedDocument:
    return ExtractedDocument(
        paper_id="W123",
        title="Sample Paper",
        doi="10.1234/example",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=Path("sample.pdf"),
        pages=[
            ExtractedPage(
                page_number=1,
                text="First paragraph.\n\nSecond paragraph.",
            ),
            ExtractedPage(
                page_number=2,
                text="Third paragraph.",
            ),
        ],
    )


def test_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        TextChunker(chunk_size=0)


def test_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        TextChunker(chunk_size=100, overlap=-1)


def test_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(ValueError, match="smaller"):
        TextChunker(chunk_size=100, overlap=100)


def test_chunks_short_page(
    document: ExtractedDocument,
) -> None:
    chunks = TextChunker().chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."
    assert chunks[1].text == "Third paragraph."


def test_preserves_page_provenance(
    document: ExtractedDocument,
) -> None:
    chunks = TextChunker().chunk_document(document)

    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[0].paper_id == "W123"
    assert chunks[0].doi == "10.1234/example"


def test_chunks_long_text() -> None:
    text = "word " * 500

    document = ExtractedDocument(
        paper_id="W123",
        title="Long Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=Path("sample.pdf"),
        pages=[
            ExtractedPage(page_number=1, text=text),
        ],
    )

    chunks = TextChunker(
        chunk_size=100,
        overlap=20,
    ).chunk_document(document)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_preserves_character_offsets() -> None:
    text = "ResearchPilot is an AI research assistant."

    document = ExtractedDocument(
        paper_id="W123",
        title="Sample Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=Path("sample.pdf"),
        pages=[
            ExtractedPage(page_number=1, text=text),
        ],
    )

    chunks = TextChunker(
        chunk_size=100,
        overlap=10,
    ).chunk_document(document)

    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)
    assert text[chunks[0].start_char:chunks[0].end_char] == chunks[0].text


def test_empty_page_produces_no_chunks() -> None:
    document = ExtractedDocument(
        paper_id="W123",
        title="Empty Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=Path("sample.pdf"),
        pages=[
            ExtractedPage(page_number=1, text=""),
        ],
    )

    chunks = TextChunker().chunk_document(document)

    assert chunks == []

def test_chunks_start_at_word_boundaries() -> None:
    text = (
        "alpha beta gamma delta epsilon zeta "
        "eta theta iota kappa lambda mu"
    )

    document = ExtractedDocument(
        paper_id="W123",
        title="Boundary Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=Path("sample.pdf"),
        pages=[
            ExtractedPage(page_number=1, text=text),
        ],
    )

    chunks = TextChunker(
        chunk_size=30,
        overlap=10,
    ).chunk_document(document)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.start_char == 0 or text[
            chunk.start_char - 1
        ].isspace(), (
            f"Chunk starts in the middle of a word: "
            f"{chunk.text!r}"
        )


def test_overlap_preserves_context() -> None:
    text = (
        "alpha beta gamma delta epsilon zeta "
        "eta theta iota kappa lambda mu"
    )

    document = ExtractedDocument(
        paper_id="W123",
        title="Overlap Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=Path("sample.pdf"),
        pages=[
            ExtractedPage(page_number=1, text=text),
        ],
    )

    chunks = TextChunker(
        chunk_size=30,
        overlap=10,
    ).chunk_document(document)

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start_char < previous.end_char
        assert all(
            current.start_char > previous.start_char
            for previous, current in zip(chunks, chunks[1:])
        )

def test_overlap_fallback_does_not_start_inside_long_word() -> None:
    long_word = "supercalifragilisticexpialidocious"
    text = f"alpha {long_word} beta gamma delta epsilon"

    document = ExtractedDocument(
        paper_id="W128",
        title="Long Word Paper",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        file_path=Path("sample.pdf"),
        pages=[
            ExtractedPage(page_number=1, text=text),
        ],
    )

    chunks = TextChunker(
        chunk_size=40,
        overlap=10,
    ).chunk_document(document)

    assert len(chunks) > 1

    for chunk in chunks[1:]:
        assert chunk.start_char == 0 or text[
            chunk.start_char - 1
        ].isspace(), (
            f"Chunk starts inside a word: {chunk.text!r}"
        )
