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
You are a research planning assistant.

Your task is to analyze the user's research question and create
a structured research plan.

Instructions:
1. Identify the main research objective.
2. Break the question into relevant sub-questions.
3. Generate focused search queries.
4. Consider both web and academic sources.
5. Select source types based on the research requirements.
6. Do not invent research findings or claim that searches
   have already been performed.
"""

def generate_research_plan(question: str) -> ResearchPlan:
    """Generate and validate a structured research plan."""

    if not question.strip():
        raise ValueError("Research question cannot be empty.")

    client = create_client()

    try:
        response = client.responses.parse(
            model=get_openai_model,
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
