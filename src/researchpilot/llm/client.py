from functools import lru_cache

from openai import APIError, OpenAI
from researchpilot.llm.usage import track_usage
from researchpilot.config import (
    get_openai_api_key,
    get_openai_model,
)
from researchpilot.llm.reliability import (
    LLMEmptyResponseError,
    LLMUnexpectedResponseError,
    raise_llm_error,
)


REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2


@lru_cache(maxsize=1)
def create_client() -> OpenAI:
    """Create and cache a reusable OpenAI client."""

    return OpenAI(
        api_key=get_openai_api_key(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def generate_response(prompt: str) -> str:
    """Generate a validated text response using the OpenAI Responses API."""

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = create_client()

    try:
        response = client.responses.create(
            model=get_openai_model(),
            input=prompt,
        )

    except APIError as error:
        raise_llm_error(error)

    track_usage(response)

    output_text = getattr(response, "output_text", None)

    if not isinstance(output_text, str):
        raise LLMUnexpectedResponseError(
            "The LLM API returned an unexpected response format."
        )

    if not output_text.strip():
        raise LLMEmptyResponseError(
            "The LLM API returned an empty response."
        )

    return output_text
