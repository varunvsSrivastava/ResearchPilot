
import pytest

from researchpilot.retrieval.academic.openalex import OpenAlexProvider


def test_reconstruct_abstract():
    inverted_index = {
        "Retrieval": [0],
        "augmented": [1],
        "generation": [2],
        "improves": [3],
        "research": [4],
    }

    abstract = OpenAlexProvider._reconstruct_abstract(inverted_index)

    assert abstract == "Retrieval augmented generation improves research"


def test_reconstruct_abstract_returns_none():
    assert OpenAlexProvider._reconstruct_abstract(None) is None
    assert OpenAlexProvider._reconstruct_abstract({}) is None


def test_parse_work():
    work = {
        "id": "https://openalex.org/W123",
        "display_name": "Research Paper",
        "abstract_inverted_index": {
            "Research": [0],
            "paper": [1],
        },
        "authorships": [
            {
                "author": {
                    "display_name": "John Doe"
                }
            }
        ],
        "publication_year": 2024,
        "doi": "https://doi.org/10.1234/example",
        "cited_by_count": 10,
        "open_access": {
            "is_oa": True
        },
        "best_oa_location": {
            "pdf_url": "https://example.com/paper.pdf"
        },
    }

    paper = OpenAlexProvider._parse_work(work, "research paper")

    assert paper.paper_id == "W123"
    assert paper.title == "Research Paper"
    assert paper.abstract == "Research paper"
    assert paper.authors == ["John Doe"]
    assert paper.publication_year == 2024
    assert paper.doi == "10.1234/example"
    assert paper.citation_count == 10
    assert paper.is_open_access is True
    assert paper.pdf_url == "https://example.com/paper.pdf"


@pytest.mark.parametrize("limit", [0, 101])
def test_search_rejects_invalid_limit(limit):
    provider = OpenAlexProvider()

    with pytest.raises(ValueError):
        provider.search("research paper", limit=limit)


def test_search_rejects_empty_query():
    provider = OpenAlexProvider()

    with pytest.raises(ValueError):
        provider.search("   ")


def test_extract_pdf_urls_prioritizes_best_oa_location():
    work = {
        "best_oa_location": {
            "pdf_url": "https://example.com/best.pdf"
        },
        "locations": [
            {"pdf_url": "https://example.com/other.pdf"},
            {"pdf_url": "https://example.com/best.pdf"},
        ],
    }

    pdf_urls = OpenAlexProvider._extract_pdf_urls(work)

    assert pdf_urls == [
        "https://example.com/best.pdf",
        "https://example.com/other.pdf",
    ]


def test_extract_pdf_urls_uses_fallback_locations():
    work = {
        "best_oa_location": {},
        "locations": [
            {"pdf_url": "https://example.com/fallback.pdf"},
        ],
    }

    pdf_urls = OpenAlexProvider._extract_pdf_urls(work)

    assert pdf_urls == ["https://example.com/fallback.pdf"]


def test_extract_pdf_urls_ignores_invalid_and_duplicate_urls():
    work = {
        "best_oa_location": {
            "pdf_url": "https://example.com/paper.pdf"
        },
        "locations": [
            {"pdf_url": "https://example.com/paper.pdf"},
            {"pdf_url": ""},
            {"pdf_url": None},
            {},
            None,
        ],
    }

    pdf_urls = OpenAlexProvider._extract_pdf_urls(work)

    assert pdf_urls == ["https://example.com/paper.pdf"]


def test_extract_pdf_urls_returns_empty_when_unavailable():
    work = {
        "best_oa_location": {},
        "locations": [],
    }

    assert OpenAlexProvider._extract_pdf_urls(work) == []


def test_extract_pdf_urls_handles_missing_locations():
    assert OpenAlexProvider._extract_pdf_urls({}) == []


def test_parse_work_preserves_all_pdf_urls():
    work = {
        "id": "https://openalex.org/W123",
        "display_name": "Research Paper",
        "best_oa_location": {
            "pdf_url": "https://example.com/best.pdf"
        },
        "locations": [
            {"pdf_url": "https://example.com/other.pdf"},
        ],
    }

    paper = OpenAlexProvider._parse_work(work, "research paper")

    assert paper.pdf_url == "https://example.com/best.pdf"
    assert paper.pdf_urls == [
        "https://example.com/best.pdf",
        "https://example.com/other.pdf",
    ]
