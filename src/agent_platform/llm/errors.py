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


class LLMPromptError(LLMError):
    """Base error for prompt construction failures."""


class LLMPromptVariableError(LLMPromptError):
    """Raised when a required prompt variable is missing."""

    def __init__(
        self,
        variable_name: str,
    ) -> None:
        super().__init__(
            f"Missing required prompt variable: {variable_name}",
            retryable=False,
        )
        self.variable_name = variable_name


class LLMPromptRegistryError(LLMPromptError):
    """Base error for prompt registry failures."""


class LLMPromptAlreadyExistsError(LLMPromptRegistryError):
    """Raised when a prompt name/version is already registered."""

    def __init__(
        self,
        name: str,
        version: str,
    ) -> None:
        super().__init__(f"Prompt already registered: {name} version {version}")
        self.name = name
        self.version = version


class LLMPromptNotFoundError(LLMPromptRegistryError):
    """Raised when a requested prompt name/version is not registered."""

    def __init__(
        self,
        name: str,
        version: str,
    ) -> None:
        super().__init__(f"Prompt not found: {name} version {version}")
        self.name = name
        self.version = version


class LLMPromptActiveVersionNotSetError(LLMPromptRegistryError):
    """Raised when a prompt has no active version configured."""

    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__(f"Active prompt version is not set: {name}")
        self.name = name


class LLMPromptLifecycleError(LLMPromptRegistryError):
    """Raised when a prompt lifecycle transition is not allowed."""


class LLMPromptActiveDeprecationError(LLMPromptLifecycleError):
    """Raised when attempting to deprecate the active prompt version."""

    def __init__(
        self,
        name: str,
        version: str,
    ) -> None:
        super().__init__(f"Cannot deprecate active prompt: {name} version {version}")
        self.name = name
        self.version = version
