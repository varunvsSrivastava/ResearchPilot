from researchpilot.retrieval.models import Paper, SearchResult


def test_paper_model_creation():
    paper = Paper(
        paper_id="W123",
        provider="openalex",
        title="Retrieval Augmented Generation",
        source_url="https://example.com/paper",
        search_query="RAG",
    )

    assert paper.title == "Retrieval Augmented Generation"
    assert paper.abstract is None
    assert paper.authors == []


def test_search_result_model_creation():
    result = SearchResult(
        query="RAG",
        provider="openalex",
        papers=[],
        total_results=100,
    )

    assert result.query == "RAG"
    assert result.total_results == 100
    assert result.papers == []