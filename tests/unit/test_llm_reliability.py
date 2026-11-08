
from unittest.mock import MagicMock

import pytest

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from researchpilot.llm.reliability import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServerError,
    LLMTimeoutError,
    raise_llm_error,
)


@pytest.fixture
def mock_request():
    return MagicMock()


def create_status_error(error_class, status_code, request):
    response = MagicMock()
    response.request = request
    response.status_code = status_code

    return error_class(
        message="Simulated API failure",
        response=response,
        body=None,
    )


@pytest.mark.parametrize(
    "original_error, expected_error",
    [
        (
            lambda request: APITimeoutError(request=request),
            LLMTimeoutError,
        ),
        (
            lambda request: APIConnectionError(
                message="Connection failed",
                request=request,
            ),
            LLMConnectionError,
        ),
        (
            lambda request: create_status_error(
                RateLimitError, 429, request
            ),
            LLMRateLimitError,
        ),
        (
            lambda request: create_status_error(
                InternalServerError, 500, request
            ),
            LLMServerError,
        ),
        (
            lambda request: create_status_error(
                AuthenticationError, 401, request
            ),
            LLMAuthenticationError,
        ),
        (
            lambda request: create_status_error(
                BadRequestError, 400, request
            ),
            LLMRequestError,
        ),
        (
            lambda request: APIError(
                message="Unexpected API failure",
                request=request,
                body=None,
            ),
            LLMAPIError,
        ),
    ],
)
def test_raise_llm_error_mapping(
    mock_request,
    original_error,
    expected_error,
):
    api_error = original_error(mock_request)

    with pytest.raises(expected_error) as exc_info:
        raise_llm_error(api_error)

    assert exc_info.value.__cause__ is api_error
