
"""Evidence indexing service."""

from researchpilot.rag.embeddings.base import EmbeddingProvider
from researchpilot.rag.indexing.base import EvidenceIndexer
from researchpilot.rag.indexing.models import (
    IndexingRequest,
    IndexingResult,
)
from researchpilot.rag.vectorstore.base import VectorStore
from researchpilot.rag.vectorstore.models import VectorRecord
from researchpilot.retrieval.fulltext.models import Evidence


class EvidenceIndexingService(EvidenceIndexer):
    """Embed and index academic evidence."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store


    def index_evidence(
        self,
        evidence: list[Evidence],
        request: IndexingRequest | None = None,
    ) -> IndexingResult:
        """Embed and index unique academic evidence records."""

        request = request or IndexingRequest()
        selected_evidence = evidence

        if request.evidence_ids:
            selected_ids = set(request.evidence_ids)
            selected_evidence = [
                item
                for item in evidence
                if item.evidence_id in selected_ids
            ]

        requested_evidence_count = len(selected_evidence)

        if not selected_evidence:
            return IndexingResult()

        # Deduplicate incoming evidence while preserving its first occurrence.
        unique_evidence: list[Evidence] = []
        seen_ids: set[str] = set()

        for item in selected_evidence:
            if item.evidence_id not in seen_ids:
                unique_evidence.append(item)
                seen_ids.add(item.evidence_id)

        duplicate_count = requested_evidence_count - len(unique_evidence)
        selected_evidence = unique_evidence

        existing_ids = set(
            getattr(self.vector_store, "records", {}).keys()
        )

        skipped_count = duplicate_count

        if request.overwrite:
            existing_evidence = [
                item.evidence_id
                for item in selected_evidence
                if item.evidence_id in existing_ids
            ]

            if existing_evidence:
                self.vector_store.delete_records(existing_evidence)

        else:
            new_evidence = [
                item
                for item in selected_evidence
                if item.evidence_id not in existing_ids
            ]

            skipped_count += len(selected_evidence) - len(new_evidence)
            selected_evidence = new_evidence

        if not selected_evidence:
            return IndexingResult(
                total_evidence=requested_evidence_count,
                indexed_evidence=0,
                skipped_evidence=skipped_count,
                failed_evidence=0,
            )

        indexed_count = 0
        failed_count = 0
        errors = []

        for start in range(
            0,
            len(selected_evidence),
            request.batch_size,
        ):
            batch = selected_evidence[
                start:start + request.batch_size
            ]

            try:
                embedding_batch = self.embedding_provider.embed_texts(
                    [item.text for item in batch]
                )

                records = [
                    VectorRecord(
                        id=item.evidence_id,
                        text=item.text,
                        vector=embedding.vector,
                        metadata={
                            "paper_id": item.paper_id,
                            "title": item.title,
                            "doi": item.doi or "",
                            "source_url": item.source_url,
                            "pdf_url": item.pdf_url,
                            "page_number": item.page_number,
                            "chunk_index": item.chunk_index,
                        },
                    )
                    for item, embedding in zip(
                        batch,
                        embedding_batch.embeddings,
                    )
                ]

                self.vector_store.add_records(records)
                indexed_count += len(records)

            except Exception as exc:
                failed_count += len(batch)
                errors.append(str(exc))

        return IndexingResult(
            total_evidence=requested_evidence_count,
            indexed_evidence=indexed_count,
            skipped_evidence=skipped_count,
            failed_evidence=failed_count,
            errors=errors,
        )
