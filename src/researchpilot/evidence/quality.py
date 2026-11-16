
"""Evidence quality scoring utilities."""

import re

from researchpilot.evidence.models import EvidenceQualityScore


class EvidenceQualityEvaluator:
    """Evaluate evidence quality using deterministic heuristics."""

    def evaluate(
        self,
        content: str,
        research_question: str = "",
    ) -> EvidenceQualityScore:
        """Evaluate relevance, completeness, and clarity."""
        if not content.strip():
            raise ValueError("Evidence content cannot be empty.")

        words = content.split()
        word_count = len(words)
        sentence_count = sum(
            content.count(mark) for mark in (".", "!", "?")
        )

        completeness = min(word_count / 100.0, 1.0)

        average_sentence_length = (
            word_count / sentence_count if sentence_count else word_count
        )

        clarity = max(
            0.0,
            min(1.0, 1.0 - abs(average_sentence_length - 20) / 40),
        )

        relevance = self._calculate_relevance(
            content,
            research_question,
        )

        overall_score = (
            relevance + completeness + clarity
        ) / 3

        return EvidenceQualityScore(
            relevance=relevance,
            completeness=completeness,
            clarity=clarity,
            overall_score=overall_score,
        )

    @staticmethod
    def _calculate_relevance(
        content: str,
        research_question: str,
    ) -> float:
        """Calculate lexical query-term coverage."""
        query_terms = set(
            re.findall(r"\b\w+\b", research_question.lower())
        )

        if not query_terms:
            return 0.0

        content_terms = set(
            re.findall(r"\b\w+\b", content.lower())
        )

        return len(query_terms & content_terms) / len(query_terms)
