from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperationalObservabilitySnapshot:
    """Unified operational snapshot for the governed LLM platform."""

    health: dict[str, Any]
    runtime: dict[str, Any]
    api: dict[str, Any]
    concurrency: dict[str, Any]
    rate_limit: dict[str, Any]
    security: dict[str, Any]
    export: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable observability snapshot."""

        return {
            "health": dict(self.health),
            "runtime": dict(self.runtime),
            "api": dict(self.api),
            "concurrency": dict(self.concurrency),
            "rate_limit": dict(self.rate_limit),
            "security": dict(self.security),
            "export": dict(self.export),
        }
