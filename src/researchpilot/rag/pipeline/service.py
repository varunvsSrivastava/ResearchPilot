
"""RAG pipeline orchestration service."""

from researchpilot.rag.pipeline.context import ContextBuilder
from researchpilot.rag.pipeline.models import (
    RAGPipelineRequest,
    RAGPipelineResult,
)
from researchpilot.rag.retrieval.base import RetrievalService
from researchpilot.rag.retrieval.models import RetrievalRequest


class RAGPipelineService:
    """Orchestrate semantic retrieval and evidence context assembly."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        """Initialize the RAG pipeline service."""
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder or ContextBuilder()

    def execute(
        self,
        request: RAGPipelineRequest,
    ) -> RAGPipelineResult:
        """Retrieve evidence and assemble attributed context."""
        retrieval_request = RetrievalRequest(
            query=request.query,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

        retrieval_response = self.retrieval_service.retrieve(
            retrieval_request
        )

        context = self.context_builder.build(
            retrieval_response.results
        )

        return RAGPipelineResult(
            query=request.query,
            retrieval_response=retrieval_response,
            context=context,
            total_results=retrieval_response.total_results,
        )
