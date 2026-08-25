from dataclasses import dataclass

from agent_platform.llm.routing_policy_version import (
    RoutingPolicyVersion,
)


@dataclass(frozen=True)
class RoutingPolicyMetadata:
    """Governance metadata for an LLM routing policy."""

    name: str
    version: RoutingPolicyVersion
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate routing policy metadata."""

        if not self.name.strip():
            raise ValueError("routing policy name must not be empty")

    @property
    def identifier(self) -> str:
        """Return a stable human-readable policy identifier."""

        return f"{self.name}@{self.version}"
