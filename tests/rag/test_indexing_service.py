
"""Tests for evidence indexing service."""

import pytest

from researchpilot.rag.embeddings.models import (
    EmbeddingBatch,
    EmbeddingVector,
)
from researchpilot.rag.indexing import (
    EvidenceIndexingService,
    IndexingRequest,
)
from researchpilot.rag.vectorstore.models import VectorRecord
from researchpilot.retrieval.fulltext.models import Evidence


class FakeEmbeddingProvider:
    """Fake embedding provider for tests."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        self.calls.append(texts)

        if self.fail:
            raise RuntimeError("Embedding failed")

        embeddings = [
            EmbeddingVector(
                text=text,
                vector=[0.1, 0.2, 0.3],
                dimensions=3,
                model="test-model",
            )
            for text in texts
        ]

        return EmbeddingBatch(
            embeddings=embeddings,
            model="test-model",
            total_embeddings=len(embeddings),
        )


class FakeVectorStore:
    """Fake vector store for tests."""

    def __init__(self, fail: bool = False):
        self.records: dict[str, VectorRecord] = {}
        self.fail = fail
        self.deleted_ids = []

    def add_records(self, records: list[VectorRecord]) -> None:
        if self.fail:
            raise RuntimeError("Vector store failed")

        for record in records:
            self.records[record.id] = record

    def delete_records(self, record_ids: list[str]) -> None:
        self.deleted_ids.extend(record_ids)

        for record_id in record_ids:
            self.records.pop(record_id, None)



def create_evidence(
    evidence_id: str,
    text: str = "Sample evidence text",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        paper_id="paper-1",
        title="Sample Paper",
        doi="10.1234/test",
        source_url="https://example.com/source",
        pdf_url="https://example.com/paper.pdf",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
    )



@pytest.fixture
def evidence_list() -> list[Evidence]:
    return [
        create_evidence("evidence-1", "First evidence"),
        create_evidence("evidence-2", "Second evidence"),
        create_evidence("evidence-3", "Third evidence"),
    ]


@pytest.fixture
def dependencies():
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()

    service = EvidenceIndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    return service, embedding_provider, vector_store


def test_index_evidence_success(evidence_list, dependencies):
    service, embedding_provider, vector_store = dependencies

    result = service.index_evidence(evidence_list)

    assert result.indexed_evidence == 3
    assert result.failed_evidence == 0
    assert result.total_evidence == 3
    assert result.is_successful

    assert len(vector_store.records) == 3
    assert len(embedding_provider.calls) == 1


def test_index_evidence_batches(evidence_list, dependencies):
    service, embedding_provider, vector_store = dependencies

    request = IndexingRequest(batch_size=2)

    result = service.index_evidence(evidence_list, request)

    assert result.indexed_evidence == 3
    assert result.failed_evidence == 0
    assert len(embedding_provider.calls) == 2
    assert len(vector_store.records) == 3


def test_index_evidence_filters_ids(evidence_list, dependencies):
    service, embedding_provider, vector_store = dependencies

    request = IndexingRequest(
        evidence_ids=["evidence-2"],
    )

    result = service.index_evidence(evidence_list, request)

    assert result.indexed_evidence == 1
    assert result.failed_evidence == 0
    assert result.total_evidence == 1
    assert list(vector_store.records) == ["evidence-2"]
    assert embedding_provider.calls == [["Second evidence"]]


def test_index_evidence_skips_existing_records(
    evidence_list,
    dependencies,
):
    service, embedding_provider, vector_store = dependencies

    vector_store.records["evidence-1"] = VectorRecord(
        id="evidence-1",
        text="Existing evidence",
        vector=[0.1, 0.2, 0.3],
        metadata={},
    )

    result = service.index_evidence(evidence_list)

    assert result.indexed_evidence == 2
    assert result.failed_evidence == 0
    assert "evidence-1" not in [
        record.id for record in vector_store.records.values()
        if record.text == "First evidence"
    ]
    assert len(vector_store.records) == 3


def test_index_evidence_overwrites_existing_records(
    evidence_list,
    dependencies,
):
    service, embedding_provider, vector_store = dependencies

    vector_store.records["evidence-1"] = VectorRecord(
        id="evidence-1",
        text="Old evidence",
        vector=[0.1, 0.2, 0.3],
        metadata={},
    )

    request = IndexingRequest(overwrite=True)

    result = service.index_evidence(evidence_list, request)

    assert result.indexed_evidence == 3
    assert result.failed_evidence == 0
    assert vector_store.records["evidence-1"].text == "First evidence"
    assert "evidence-1" in vector_store.deleted_ids


def test_index_evidence_handles_embedding_failure(evidence_list):
    embedding_provider = FakeEmbeddingProvider(fail=True)
    vector_store = FakeVectorStore()

    service = EvidenceIndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    result = service.index_evidence(evidence_list)

    assert result.indexed_evidence == 0
    assert result.failed_evidence == 3
    assert len(result.errors) == 1
    assert not result.is_successful
    assert vector_store.records == {}


def test_index_evidence_handles_vector_store_failure(evidence_list):
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore(fail=True)

    service = EvidenceIndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    result = service.index_evidence(evidence_list)

    assert result.indexed_evidence == 0
    assert result.failed_evidence == 3
    assert len(result.errors) == 1
    assert not result.is_successful


def test_index_evidence_empty_list(dependencies):
    service, embedding_provider, vector_store = dependencies

    result = service.index_evidence([])

    assert result.total_evidence == 0
    assert result.indexed_evidence == 0
    assert result.failed_evidence == 0
    assert result.is_successful
    assert embedding_provider.calls == []
    assert vector_store.records == {}


def test_index_evidence_all_existing_records(
    evidence_list,
    dependencies,
):
    service, embedding_provider, vector_store = dependencies

    for item in evidence_list:
        vector_store.records[item.evidence_id] = VectorRecord(
            id=item.evidence_id,
            text=item.text,
            vector=[0.1, 0.2, 0.3],
            metadata={},
        )

    result = service.index_evidence(evidence_list)

    assert result.indexed_evidence == 0
    assert result.failed_evidence == 0
    assert embedding_provider.calls == []
    assert len(vector_store.records) == 3
