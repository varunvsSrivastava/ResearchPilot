
from typing import NoReturn

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)


class LLMError(RuntimeError):
    """Base exception for ResearchPilot LLM failures."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""


class LLMConnectionError(LLMError):
    """Raised when an LLM connection fails."""


class LLMRateLimitError(LLMError):
    """Raised when the API rate limit is exceeded."""


class LLMServerError(LLMError):
    """Raised when the API server encounters an error."""


class LLMAuthenticationError(LLMError):
    """Raised when API authentication fails."""


class LLMRequestError(LLMError):
    """Raised when an API request is invalid."""


class LLMAPIError(LLMError):
    """Raised for other API-related failures."""


class LLMEmptyResponseError(LLMError):
    """Raised when the LLM returns empty or whitespace-only text."""


class LLMUnexpectedResponseError(LLMError):
    """Raised when the LLM returns an unexpected response format."""


def raise_llm_error(error: APIError) -> NoReturn:
    """Convert OpenAI API exceptions into ResearchPilot exceptions."""

    if isinstance(error, APITimeoutError):
        raise LLMTimeoutError(
            "The LLM request timed out."
        ) from error

    if isinstance(error, RateLimitError):
        raise LLMRateLimitError(
            "The LLM API rate limit was exceeded."
        ) from error

    if isinstance(error, InternalServerError):
        raise LLMServerError(
            "The LLM API server encountered an error."
        ) from error

    if isinstance(error, AuthenticationError):
        raise LLMAuthenticationError(
            "LLM API authentication failed. Check your API key."
        ) from error

    if isinstance(error, BadRequestError):
        raise LLMRequestError(
            "The LLM API rejected the request."
        ) from error

    if isinstance(error, APIConnectionError):
        raise LLMConnectionError(
            "Unable to establish a connection to the LLM API."
        ) from error

    raise LLMAPIError(
        "An unexpected LLM API error occurred."
    ) from error
