
"""RAG pipeline orchestration service."""

from researchpilot.rag.pipeline.models import (
    RAGPipelineRequest,
    RAGPipelineResult,
)
from researchpilot.rag.retrieval.base import RetrievalService
from researchpilot.rag.retrieval.models import RetrievalRequest


class RAGPipelineService:
    """Orchestrate semantic retrieval and evidence context assembly."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

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

        context_sections = []

        for index, result in enumerate(
            retrieval_response.results,
            start=1,
        ):
            metadata = result.record.metadata

            source_lines = [
                f"[Source {index}]",
                f"Evidence ID: {result.record.id}",
            ]

            for field, label in (
                ("paper_id", "Paper ID"),
                ("title", "Title"),
                ("doi", "DOI"),
                ("page_number", "Page"),
            ):
                value = metadata.get(field)
                if value is not None:
                    source_lines.append(f"{label}: {value}")

            source_lines.append("Evidence:")
            source_lines.append(result.record.text)

            context_sections.append("\n".join(source_lines))

        context = "\n\n".join(context_sections)

        return RAGPipelineResult(
            query=request.query,
            retrieval_response=retrieval_response,
            context=context,
            total_results=retrieval_response.total_results,
        )
