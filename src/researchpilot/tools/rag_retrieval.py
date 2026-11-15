
"""RAG retrieval tool for research agent."""

from pydantic import BaseModel, Field

from researchpilot.tools.base import ResearchTool


class RAGRetrievalInput(BaseModel):
    """Arguments for semantic evidence retrieval."""

    query: str = Field(
        min_length=3,
        description="Research query for semantic evidence retrieval.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of evidence chunks to retrieve.",
    )


class RAGRetrievalTool(ResearchTool):
    """Retrieve relevant evidence using semantic search."""

    name = "rag_retrieval"
    description = (
        "Retrieve relevant research evidence from the indexed knowledge base."
    )
    args_schema = RAGRetrievalInput

    def __init__(self, retrieval_service) -> None:
        self.retrieval_service = retrieval_service

    def execute(self, arguments: RAGRetrievalInput):
        """Execute semantic evidence retrieval."""
        from researchpilot.rag.retrieval.models import RetrievalRequest

        request = RetrievalRequest(
            query=arguments.query,
            top_k=arguments.top_k,
        )

        return self.retrieval_service.retrieve(request)
