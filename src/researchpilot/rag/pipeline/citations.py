
"""Citation validation for generated research answers."""

import re


class CitationValidator:
    """Validate source citations against retrieved evidence."""

    CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")

    def extract_citations(self, answer: str) -> set[int]:
        """Extract source numbers cited in an answer."""
        return {
            int(match)
            for match in self.CITATION_PATTERN.findall(answer)
        }

    def extract_available_sources(self, context: str) -> set[int]:
        """Extract source numbers available in retrieved context."""
        return {
            int(match)
            for match in self.CITATION_PATTERN.findall(context)
        }

    def validate(self, answer: str, context: str) -> bool:
        """Return whether every answer citation exists in the context."""
        answer_citations = self.extract_citations(answer)
        available_sources = self.extract_available_sources(context)

        return answer_citations.issubset(available_sources)
