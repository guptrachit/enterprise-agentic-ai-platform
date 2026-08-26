from dataclasses import dataclass
from typing import Protocol

from agent_platform.config import Settings


class ProductionReadinessError(RuntimeError):
    """Raised when production startup readiness checks fail."""


class RuntimeHealthSnapshot(Protocol):
    """Minimal runtime health snapshot contract used at startup."""

    def to_dict(
        self,
    ) -> dict[str, object]:
        """Return JSON-serializable runtime health."""


class RuntimeHealthProvider(Protocol):
    """Minimal runtime health provider used at startup."""

    def health_snapshot(
        self,
    ) -> RuntimeHealthSnapshot:
        """Return current runtime health."""


@dataclass(frozen=True)
class ProductionReadinessValidator:
    """Validate production startup configuration."""

    settings: Settings

    def validate(self) -> None:
        """Fail startup when production configuration is unsafe."""

        if self.settings.app_env.lower() != "production":
            return

        if not self.settings.llm_api_authentication_required:
            raise ProductionReadinessError(
                "Production requires LLM API authentication."
            )

        if not self.settings.llm_api_cors_allowed_origins:
            raise ProductionReadinessError(
                "Production requires at least one explicit CORS origin."
            )

        if "*" in self.settings.llm_api_cors_allowed_origins:
            raise ProductionReadinessError(
                "Wildcard CORS origin is not allowed in production."
            )

        if self.settings.llm_api_max_request_body_bytes <= 0:
            raise ProductionReadinessError(
                "Production request body limit must be greater than 0."
            )

        if not self.settings.llm_routing_policy_name.strip():
            raise ProductionReadinessError(
                "Production routing policy name must not be empty."
            )


def validate_production_readiness(
    settings: Settings,
) -> None:
    """Validate application production configuration readiness."""

    ProductionReadinessValidator(settings=settings).validate()


def validate_runtime_readiness(
    runtime: RuntimeHealthProvider,
) -> None:
    """Fail startup when governed runtime cannot resolve."""

    snapshot = runtime.health_snapshot().to_dict()

    if not bool(
        snapshot.get(
            "resolved",
            False,
        )
    ):
        raise ProductionReadinessError("Governed LLM runtime is not ready.")
