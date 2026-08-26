from agent_platform.llm.api_errors import (
    map_llm_exception,
)
from agent_platform.llm.errors import (
    LLMInvalidRequestError,
    LLMModelNotFoundError,
    LLMTransientError,
)


def test_maps_invalid_request() -> None:
    result = map_llm_exception(LLMInvalidRequestError())

    assert result.status_code == 400
    assert result.code == "invalid_llm_request"


def test_maps_model_not_found() -> None:
    result = map_llm_exception(LLMModelNotFoundError("missing-model"))

    assert result.status_code == 503
    assert result.code == "llm_model_unavailable"


def test_maps_transient_failure() -> None:
    result = map_llm_exception(LLMTransientError())

    assert result.status_code == 503

    assert result.code == ("llm_temporarily_unavailable")


def test_maps_missing_active_policy() -> None:
    result = map_llm_exception(LookupError("No active routing policy found."))

    assert result.status_code == 503

    assert result.code == ("routing_policy_unavailable")


def test_maps_unexpected_failure() -> None:
    result = map_llm_exception(RuntimeError("internal detail"))

    assert result.status_code == 500
    assert result.code == "llm_internal_error"

    assert result.message == ("An unexpected LLM runtime error occurred.")


def test_maps_request_timeout() -> None:
    from agent_platform.llm.api_errors import (
        LLMRequestTimeoutError,
    )

    result = map_llm_exception(LLMRequestTimeoutError())

    assert result.status_code == 504
    assert result.code == "llm_request_timeout"

    assert result.message == ("LLM request exceeded the configured timeout.")


def test_maps_capacity_exceeded() -> None:
    from agent_platform.llm.api_guardrails import (
        LLMCapacityExceededError,
    )

    result = map_llm_exception(LLMCapacityExceededError())

    assert result.status_code == 503
    assert result.code == "llm_capacity_exceeded"

    assert result.message == ("LLM service is currently at request capacity.")


def test_maps_rate_limit_exceeded() -> None:
    from agent_platform.llm.api_rate_limit import (
        LLMRateLimitExceededError,
    )

    result = map_llm_exception(LLMRateLimitExceededError())

    assert result.status_code == 429
    assert result.code == "llm_rate_limit_exceeded"

    assert result.message == ("LLM API request rate limit exceeded.")


def test_maps_authentication_required() -> None:
    from agent_platform.security.auth_policy import (
        AuthenticationRequiredError,
    )

    result = map_llm_exception(AuthenticationRequiredError())

    assert result.status_code == 401
    assert result.code == "authentication_required"

    assert result.message == ("Authentication is required.")


def test_maps_authorization_denied() -> None:
    from agent_platform.security.authorization import (
        AuthorizationDeniedError,
    )

    result = map_llm_exception(AuthorizationDeniedError())

    assert result.status_code == 403
    assert result.code == "authorization_denied"

    assert result.message == (
        "The authenticated identity is not authorized to perform this operation."
    )


def test_maps_unsupported_media_type() -> None:
    from agent_platform.security.content_type_policy import (
        UnsupportedMediaTypeError,
    )

    result = map_llm_exception(UnsupportedMediaTypeError())

    assert result.status_code == 415
    assert result.code == "unsupported_media_type"

    assert result.message == ("Content-Type must be application/json.")
