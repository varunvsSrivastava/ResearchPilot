
import pytest

from researchpilot.retrieval.fulltext.evidence import EvidenceGenerator
from researchpilot.retrieval.fulltext.models import TextChunk


@pytest.fixture
def text_chunk() -> TextChunk:
    return TextChunk(
        paper_id="W123",
        title="Sample Research Paper",
        doi="10.1234/example",
        source_url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        page_number=3,
        chunk_index=0,
        text="ResearchPilot is an AI research assistant.",
        start_char=0,
        end_char=42,
    )


def test_generates_evidence_from_chunk(
    text_chunk: TextChunk,
) -> None:
    evidence = EvidenceGenerator().generate(text_chunk)

    assert evidence.paper_id == text_chunk.paper_id
    assert evidence.title == text_chunk.title
    assert evidence.text == text_chunk.text
    assert evidence.page_number == 3
    assert evidence.chunk_index == 0


def test_preserves_source_metadata(
    text_chunk: TextChunk,
) -> None:
    evidence = EvidenceGenerator().generate(text_chunk)

    assert evidence.doi == text_chunk.doi
    assert evidence.source_url == text_chunk.source_url
    assert evidence.pdf_url == text_chunk.pdf_url


def test_preserves_character_offsets(
    text_chunk: TextChunk,
) -> None:
    evidence = EvidenceGenerator().generate(text_chunk)

    assert evidence.start_char == text_chunk.start_char
    assert evidence.end_char == text_chunk.end_char


def test_generates_deterministic_evidence_id(
    text_chunk: TextChunk,
) -> None:
    generator = EvidenceGenerator()

    first = generator.generate(text_chunk)
    second = generator.generate(text_chunk)

    assert first.evidence_id == second.evidence_id


def test_evidence_id_changes_for_different_chunks(
    text_chunk: TextChunk,
) -> None:
    generator = EvidenceGenerator()

    first = generator.generate(text_chunk)

    second_chunk = text_chunk.model_copy(
        update={
            "page_number": 4,
            "chunk_index": 1,
        }
    )
    second = generator.generate(second_chunk)

    assert first.evidence_id != second.evidence_id


def test_generates_evidence_for_multiple_chunks(
    text_chunk: TextChunk,
) -> None:
    second_chunk = text_chunk.model_copy(
        update={
            "page_number": 4,
            "chunk_index": 1,
            "text": "Second evidence chunk.",
            "start_char": 0,
            "end_char": 22,
        }
    )

    evidence = EvidenceGenerator().generate_many(
        [text_chunk, second_chunk]
    )

    assert len(evidence) == 2
    assert evidence[0].page_number == 3
    assert evidence[1].page_number == 4


def test_generate_many_returns_empty_for_empty_input() -> None:
    evidence = EvidenceGenerator().generate_many([])

    assert evidence == []


def test_evidence_rejects_empty_text(
    text_chunk: TextChunk,
) -> None:
    invalid_chunk = text_chunk.model_copy(
        update={"text": ""}
    )

    with pytest.raises(ValueError):
        EvidenceGenerator().generate(invalid_chunk)
