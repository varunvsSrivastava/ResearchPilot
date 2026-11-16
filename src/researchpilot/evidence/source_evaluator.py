
"""Evaluate credibility of academic sources."""

from researchpilot.evidence.source_credibility import (
    SourceCredibilityEvaluator,
)
from researchpilot.evidence.source_models import EvaluatedSource
from researchpilot.retrieval.models import Paper


class SourceEvaluator:
    """Evaluate credibility of multiple academic sources."""

    def __init__(self) -> None:
        self._credibility_evaluator = SourceCredibilityEvaluator()

    def evaluate(
        self,
        papers: list[Paper],
    ) -> list[EvaluatedSource]:
        """Evaluate papers and return their credibility assessments."""

        evaluated_sources = []

        for paper in papers:
            credibility = self._credibility_evaluator.evaluate(paper)

            evaluated_sources.append(
                EvaluatedSource(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    provider=paper.provider,
                    credibility=credibility,
                )
            )

        return evaluated_sources
