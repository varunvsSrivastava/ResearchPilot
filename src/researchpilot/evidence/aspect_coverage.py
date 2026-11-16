
"""Research aspect coverage assessment utilities."""

import re


class AspectCoverageAssessor:
    """Assess research aspects using lexical evidence matching."""

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "their", "this", "to", "was", "were", "what", "which",
        "with",
    }

    def __init__(self, minimum_match_ratio: float = 0.5) -> None:
        """Initialize the assessor with a minimum lexical match ratio."""
        if not 0.0 < minimum_match_ratio <= 1.0:
            raise ValueError(
                "Minimum match ratio must be between 0 and 1."
            )

        self.minimum_match_ratio = minimum_match_ratio

    def assess(
        self,
        aspects: list[str],
        evidence_texts: list[str],
    ) -> list[str]:
        """Return aspects supported by sufficient lexical evidence."""
        evidence_tokens = set()

        for text in evidence_texts:
            evidence_tokens.update(self._tokenize(text))

        covered_aspects = []

        for aspect in aspects:
            aspect_tokens = self._tokenize(aspect)

            if not aspect_tokens:
                continue

            match_ratio = (
                len(aspect_tokens & evidence_tokens)
                / len(aspect_tokens)
            )

            if match_ratio >= self.minimum_match_ratio:
                covered_aspects.append(aspect)

        return covered_aspects

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        """Extract meaningful lowercase lexical tokens."""
        tokens = set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))

        return tokens - cls._STOPWORDS
