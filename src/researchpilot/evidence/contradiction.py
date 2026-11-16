
"""Evidence contradiction detection utilities."""

import re

from researchpilot.evidence.contradiction_models import (
    EvidenceContradiction,
)
from researchpilot.evidence.models import Evidence


class ContradictionDetector:
    """Detect potential contradictions between evidence items."""

    NEGATION_TERMS = {
        "not",
        "no",
        "never",
        "none",
        "neither",
        "without",
        "cannot",
        "can't",
        "doesn't",
        "don't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
    }

    def detect(
        self,
        evidence_items: list[Evidence],
    ) -> list[EvidenceContradiction]:
        """Identify potential contradictions between evidence items."""

        contradictions = []

        for index, first in enumerate(evidence_items):
            for second in evidence_items[index + 1:]:
                contradiction = self._compare(first, second)

                if contradiction is not None:
                    contradictions.append(contradiction)

        return contradictions

    def _compare(
        self,
        first: Evidence,
        second: Evidence,
    ) -> EvidenceContradiction | None:
        """Compare two evidence items for potential contradictions."""

        first_tokens = self._tokenize(first.content)
        second_tokens = self._tokenize(second.content)

        shared_tokens = first_tokens & second_tokens
        if len(shared_tokens) < 3:
            return None

        first_negated = bool(first_tokens & self.NEGATION_TERMS)
        second_negated = bool(second_tokens & self.NEGATION_TERMS)

        if first_negated == second_negated:
            return None

        overlap_score = len(shared_tokens) / min(
            len(first_tokens),
            len(second_tokens),
        )

        contradiction_score = min(1.0, overlap_score)

        return EvidenceContradiction(
            first_source_id=first.source_id,
            second_source_id=second.source_id,
            first_claim=first.content,
            second_claim=second.content,
            contradiction_score=contradiction_score,
            explanation=(
                "Evidence statements share substantial lexical content "
                "but differ in negation."
            ),
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Normalize text into lowercase lexical tokens."""

        return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
