class LLMError(Exception):
    """Base exception for LLM failures."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""

    def __init__(
        self,
        message: str = "LLM request timed out.",
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            retry_after_seconds=retry_after_seconds,
        )


class LLMRateLimitError(LLMError):
    """Raised when the provider rate-limits the request."""

    def __init__(
        self,
        message: str = "LLM rate limit exceeded.",
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            retry_after_seconds=retry_after_seconds,
        )


class LLMTransientError(LLMError):
    """Raised for retryable transient provider failures."""

    def __init__(
        self,
        message: str = "LLM provider experienced a transient failure.",
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(
            message,
            retryable=True,
            retry_after_seconds=retry_after_seconds,
        )


class LLMConfigurationError(LLMError):
    """Raised when the LLM configuration is invalid."""


class LLMInvalidRequestError(LLMError):
    """Raised when the request sent to the provider is invalid."""

    def __init__(
        self,
        message: str = "LLM request is invalid.",
    ) -> None:
        super().__init__(message)


class LLMRetryBudgetExceededError(LLMError):
    """Raised when the retry time budget is exhausted."""

    def __init__(
        self,
        message: str = "LLM retry budget exceeded.",
    ) -> None:
        super().__init__(
            message,
            retryable=False,
        )


class LLMStructuredOutputError(LLMError):
    """Base error for invalid structured LLM output."""


class LLMStructuredValidationError(LLMStructuredOutputError):
    """Raised when structured output fails schema validation."""

    def __init__(
        self,
        message: str = "LLM structured response failed validation.",
    ) -> None:
        super().__init__(
            message,
            retryable=False,
        )


class LLMStructuredParseError(LLMStructuredOutputError):
    """Raised when no usable structured response can be parsed."""

    def __init__(
        self,
        message: str = "LLM structured response could not be parsed.",
    ) -> None:
        super().__init__(
            message,
            retryable=False,
        )


class LLMRefusalError(LLMStructuredOutputError):
    """Raised when the model refuses to produce the requested output."""

    def __init__(
        self,
        message: str = "LLM refused the request.",
    ) -> None:
        super().__init__(
            message,
            retryable=False,
        )
