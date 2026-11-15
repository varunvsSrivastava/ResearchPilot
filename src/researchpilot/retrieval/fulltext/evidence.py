
"""Evidence generation from academic text chunks."""

import hashlib

from researchpilot.retrieval.fulltext.models import Evidence, TextChunk


class EvidenceGenerator:
    """Generate traceable evidence records from text chunks."""

    def generate(self, chunk: TextChunk) -> Evidence:
        """Convert a text chunk into a structured evidence record."""
        evidence_id = self._generate_evidence_id(chunk)

        return Evidence(
            evidence_id=evidence_id,
            paper_id=chunk.paper_id,
            title=chunk.title,
            doi=chunk.doi,
            source_url=chunk.source_url,
            pdf_url=chunk.pdf_url,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )

    def generate_many(
        self,
        chunks: list[TextChunk],
    ) -> list[Evidence]:
        """Generate evidence records for multiple chunks."""
        return [self.generate(chunk) for chunk in chunks]

    @staticmethod
    def _generate_evidence_id(chunk: TextChunk) -> str:
        """Create a deterministic evidence ID from chunk provenance."""
        identity = (
            f"{chunk.paper_id}|"
            f"{chunk.page_number}|"
            f"{chunk.chunk_index}|"
            f"{chunk.start_char}|"
            f"{chunk.end_char}|"
            f"{chunk.text}"
        )

        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        return f"ev_{digest[:20]}"
