
"""Generation service for evidence-grounded research answers."""

from researchpilot.llm.client import generate_response
from researchpilot.rag.pipeline.citations import CitationValidator
from researchpilot.rag.pipeline.prompt import ResearchPromptBuilder


class AnswerGenerationService:
    """Generate research answers using retrieved evidence."""

    INSUFFICIENT_EVIDENCE_RESPONSE = (
        "Insufficient evidence was retrieved to answer this research "
        "question reliably. Please refine the query or provide additional "
        "research sources."
    )

    INVALID_CITATION_RESPONSE = (
        "The generated answer contains citations to unavailable sources. "
        "Please retry answer generation."
    )

    def __init__(
        self,
        prompt_builder: ResearchPromptBuilder | None = None,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        """Initialize the answer generation service."""
        self.prompt_builder = prompt_builder or ResearchPromptBuilder()
        self.citation_validator = citation_validator or CitationValidator()

    def generate(self, query: str, context: str) -> str:
        """Generate an answer grounded in the supplied evidence."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not context.strip():
            return self.INSUFFICIENT_EVIDENCE_RESPONSE

        prompt = self.prompt_builder.build(
            query=query,
            context=context,
        )

        answer = generate_response(prompt)

        if not self.citation_validator.validate(answer, context):
            return self.INVALID_CITATION_RESPONSE

        return answer
