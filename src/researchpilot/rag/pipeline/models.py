
"""Models for RAG pipeline operations."""

from pydantic import BaseModel, Field

from researchpilot.rag.retrieval.models import RetrievalResponse
from researchpilot.rag.vectorstore.filters import MetadataFilter


class RAGPipelineRequest(BaseModel):
    """Represent a request to execute the RAG pipeline."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)
    metadata_filter: MetadataFilter | None = None
    generate_answer: bool = False


class RAGPipelineResult(BaseModel):
    """Represent the result of a RAG pipeline execution."""

    query: str
    retrieval_response: RetrievalResponse
    context: str = ""
    total_results: int = 0
    answer: str | None = None
