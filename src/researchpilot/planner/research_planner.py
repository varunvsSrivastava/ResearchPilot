from openai import APIError
from researchpilot.config import get_openai_model
from researchpilot.llm.client import create_client
from researchpilot.llm.reliability import (
    LLMUnexpectedResponseError,
    raise_llm_error,
)
from researchpilot.llm.usage import track_usage
from researchpilot.schemas.research_plan import ResearchPlan

SYSTEM_PROMPT = """
You are ResearchPilot, an evidence-driven research planning assistant.

Your task is to transform the user's research question into a
structured, actionable research plan.

Research Planning Instructions:

1. Research Objective
   - Identify the central objective of the research.
   - Preserve the user's original research intent.
   - Avoid introducing unrelated research topics.

2. Sub-Questions
   - Generate 2 to 6 focused sub-questions.
   - Cover distinct and relevant dimensions of the research.
   - Avoid redundant, overlapping, or overly broad questions.
   - Ensure the sub-questions collectively support the research objective.

3. Search Queries
   - Generate 2 to 8 focused search queries.
   - Each query must have a clearly defined retrieval purpose.
   - Prefer specific terminology over generic keyword combinations.
   - Cover different relevant dimensions of the research.
   - Avoid duplicate or near-duplicate queries.
   - Select the source type according to the evidence required.

4. Source Preferences
   - Prioritize academic sources for evidence-based research.
   - Prefer peer-reviewed studies, systematic reviews,
     and reputable scholarly publications where appropriate.
   - Use reliable web sources as supplementary evidence.
   - Consider official institutional sources when relevant.
   - Select source preferences according to the research topic.

5. Research Strategy
   - Describe a concise and practical approach to investigating
     the research question.
   - Explain how academic and supplementary sources should
     contribute to the investigation.
   - Identify important dimensions or comparisons to examine.
   - Avoid claiming that evidence has already been retrieved.

6. Reliability
   - Do not invent research findings, citations, or source URLs.
   - Do not claim that searches have already been performed.
   - Do not assume that every research question requires
     the same source-selection strategy.
   - Ensure the generated plan remains relevant to the
     original research question.

Return only the structured research plan matching the
provided ResearchPlan schema.
"""

def generate_research_plan(question: str) -> ResearchPlan:
    """Generate and validate a structured research plan."""

    if not question.strip():
        raise ValueError("Research question cannot be empty.")

    client = create_client()

    try:
        response = client.responses.parse(
            model=get_openai_model(),
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            text_format=ResearchPlan,
        )
    except APIError as error:
        raise_llm_error(error)

    track_usage(response)

    research_plan = getattr(response, "output_parsed", None)

    if research_plan is None:
        raise LLMUnexpectedResponseError(
            "The model did not return a valid research plan."
        )

    if not isinstance(research_plan, ResearchPlan):
        raise LLMUnexpectedResponseError(
            "The model returned an unexpected research plan format."
        )

    return research_plan
