
"""Context construction for RAG pipeline."""

from researchpilot.rag.vectorstore.models import VectorSearchResult


class ContextBuilder:
    """Build source-attributed context from retrieved evidence."""

    METADATA_FIELDS = (
        ("paper_id", "Paper ID"),
        ("title", "Title"),
        ("doi", "DOI"),
        ("page_number", "Page"),
    )

    def __init__(self, max_context_chars: int = 12000) -> None:
        """Initialize the context builder."""
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be positive")

        self.max_context_chars = max_context_chars

    def build(
        self,
        results: list[VectorSearchResult],
    ) -> str:
        """Build context while preserving order and respecting limits."""
        context_sections = []
        seen_evidence_ids = set()
        current_length = 0

        for result in results:
            record = result.record

            if not record.text.strip():
                continue

            if record.id in seen_evidence_ids:
                continue

            source_lines = [
                f"[Source {len(context_sections) + 1}]",
                f"Evidence ID: {record.id}",
            ]

            for field, label in self.METADATA_FIELDS:
                value = record.metadata.get(field)

                if value is not None:
                    source_lines.append(f"{label}: {value}")

            source_lines.extend(
                [
                    "Evidence:",
                    record.text,
                ]
            )

            section = "\n".join(source_lines)

            separator_length = 2 if context_sections else 0
            required_length = current_length + separator_length + len(section)

            if required_length > self.max_context_chars:
                break

            seen_evidence_ids.add(record.id)
            context_sections.append(section)
            current_length = required_length

        return "\n\n".join(context_sections)
