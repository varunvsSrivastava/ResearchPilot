
"""Dependency construction for the ResearchPilot API."""

import os
from functools import lru_cache
from pathlib import Path

from researchpilot.agent.orchestrator import ResearchOrchestrator
from researchpilot.config import (
    PROJECT_ROOT,
    get_openai_embedding_dimensions,
)
from researchpilot.generation.report_pipeline import ResearchReportPipeline
from researchpilot.rag.embeddings.openai_embeddings import (
    OpenAIEmbeddingProvider,
)
from researchpilot.rag.indexing.service import EvidenceIndexingService
from researchpilot.rag.pipeline.service import RAGPipelineService
from researchpilot.rag.retrieval.service import SemanticRetrievalService
from researchpilot.rag.vectorstore.faiss_store import FAISSVectorStore
from researchpilot.retrieval.academic.openalex import OpenAlexProvider
from researchpilot.retrieval.fulltext.service import FullTextAcquisitionService
from researchpilot.retrieval.pipeline import RetrievalPipeline
from researchpilot.retrieval.services.academic_search import (
    AcademicSearchService,
)
from researchpilot.tools.registry import ToolRegistry


class PersistentEvidenceIndexingService:
    """Persist FAISS data after evidence indexing."""

    def __init__(
        self,
        indexing_service: EvidenceIndexingService,
        vector_store: FAISSVectorStore,
        storage_path: Path,
    ) -> None:
        self.indexing_service = indexing_service
        self.vector_store = vector_store
        self.storage_path = storage_path

    def index_evidence(self, evidence):
        """Index evidence and persist the updated vector store."""
        result = self.indexing_service.index_evidence(evidence)

        # Preserve successfully indexed records, including partial batches.
        self.vector_store.save(str(self.storage_path))

        return result


def _get_vector_storage_path() -> Path:
    """Resolve the configured FAISS storage directory."""
    configured_path = Path(
        os.getenv("FAISS_STORAGE_DIR", "data/vectorstore")
    )

    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path

    return configured_path.resolve()


def _build_vector_store() -> FAISSVectorStore:
    """Create or restore the persistent FAISS vector store."""
    dimensions = get_openai_embedding_dimensions()
    storage_path = _get_vector_storage_path()

    index_path = storage_path / "index.faiss"
    metadata_path = storage_path / "metadata.json"

    vector_store = FAISSVectorStore(dimensions=dimensions)

    if index_path.exists() != metadata_path.exists():
        raise RuntimeError(
            "FAISS storage is incomplete. Both index.faiss and "
            "metadata.json must exist."
        )

    if index_path.exists():
        vector_store.load(str(storage_path))

        if vector_store.dimensions != dimensions:
            raise ValueError(
                "Persisted FAISS dimensions do not match the "
                "configured embedding dimensions."
            )

    return vector_store


@lru_cache(maxsize=1)
def get_research_orchestrator() -> ResearchOrchestrator:
    """Build and cache the complete research service container."""

    # Shared embedding provider and vector store.
    embedding_provider = OpenAIEmbeddingProvider()
    vector_store = _build_vector_store()

    # Academic retrieval.
    academic_provider = OpenAlexProvider()
    academic_search_service = AcademicSearchService(
        provider=academic_provider,
    )

    fulltext_service = FullTextAcquisitionService()

    retrieval_pipeline = RetrievalPipeline(
        academic_search_service=academic_search_service,
        fulltext_service=fulltext_service,
    )

    # Evidence indexing and semantic retrieval share the same store.
    indexing_service = EvidenceIndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    persistent_indexing_service = PersistentEvidenceIndexingService(
        indexing_service=indexing_service,
        vector_store=vector_store,
        storage_path=_get_vector_storage_path(),
    )

    semantic_retrieval_service = SemanticRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    rag_pipeline_service = RAGPipelineService(
        retrieval_service=semantic_retrieval_service,
    )

    # Research report generation.
    report_pipeline = ResearchReportPipeline()

    return ResearchOrchestrator(
        tool_registry=ToolRegistry(),
        retrieval_pipeline=retrieval_pipeline,
        indexing_service=persistent_indexing_service,
        rag_pipeline_service=rag_pipeline_service,
        report_pipeline=report_pipeline,
    )
