from dataclasses import dataclass

from fastapi import HTTPException

from agent_platform.llm.errors import (
    LLMInvalidRequestError,
    LLMModelNotFoundError,
    LLMTransientError,
)


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
