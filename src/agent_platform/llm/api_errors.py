from dataclasses import dataclass

from fastapi import HTTPException

from agent_platform.llm.api_guardrails import (
    LLMCapacityExceededError,
    PromptTooLargeError,
)
from agent_platform.llm.api_rate_limit import (
    LLMRateLimitExceededError,
)
from agent_platform.llm.errors import (
    LLMInvalidRequestError,
    LLMModelNotFoundError,
    LLMTransientError,
)
from agent_platform.security.auth_policy import (
    AuthenticationRequiredError,
)

from agent_platform.security.authorization import (
    AuthorizationDeniedError,
)


class LLMRequestTimeoutError(TimeoutError):
    """Raised when governed LLM API execution exceeds its timeout."""


@dataclass(frozen=True)
class LLMAPIError:
    """Stable API-safe representation of an LLM runtime failure."""

    status_code: int
    code: str
    message: str

    def to_http_exception(self) -> HTTPException:
        """Convert this error into a FastAPI HTTPException."""

        return HTTPException(
            status_code=self.status_code,
            detail={
                "code": self.code,
                "message": self.message,
            },
        )


def map_llm_exception(
    error: Exception,
) -> LLMAPIError:
    """Map internal governed-runtime exceptions to API-safe errors."""

    if isinstance(error, AuthenticationRequiredError):
        return LLMAPIError(
            status_code=401,
            code="authentication_required",
            message="Authentication is required.",
        )

    if isinstance(error, AuthorizationDeniedError):
        return LLMAPIError(
            status_code=403,
            code="authorization_denied",
            message="The authenticated identity is not authorized "
            "to perform this operation.",
        )

    if isinstance(error, PromptTooLargeError):
        return LLMAPIError(
            status_code=400,
            code="prompt_too_large",
            message=str(error),
        )

    if isinstance(error, LLMRateLimitExceededError):
        return LLMAPIError(
            status_code=429,
            code="llm_rate_limit_exceeded",
            message="LLM API request rate limit exceeded.",
        )

    if isinstance(error, LLMCapacityExceededError):
        return LLMAPIError(
            status_code=503,
            code="llm_capacity_exceeded",
            message="LLM service is currently at request capacity.",
        )

    if isinstance(error, LLMRequestTimeoutError):
        return LLMAPIError(
            status_code=504,
            code="llm_request_timeout",
            message="LLM request exceeded the configured timeout.",
        )

    if isinstance(error, LLMInvalidRequestError):
        return LLMAPIError(
            status_code=400,
            code="invalid_llm_request",
            message=str(error),
        )

    if isinstance(error, LLMModelNotFoundError):
        return LLMAPIError(
            status_code=503,
            code="llm_model_unavailable",
            message=str(error),
        )

    if isinstance(error, LLMTransientError):
        return LLMAPIError(
            status_code=503,
            code="llm_temporarily_unavailable",
            message="LLM service is temporarily unavailable.",
        )

    if isinstance(error, LookupError):
        return LLMAPIError(
            status_code=503,
            code="routing_policy_unavailable",
            message=str(error),
        )

    return LLMAPIError(
        status_code=500,
        code="llm_internal_error",
        message="An unexpected LLM runtime error occurred.",
    )
