from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityMetricsSnapshot:
    """Immutable snapshot of security failure metrics."""

    authentication_failures: int
    authorization_failures: int
    failure_counts: dict[str, int]

    @property
    def total_security_failures(self) -> int:
        """Return total authentication and authorization failures."""

        return self.authentication_failures + self.authorization_failures

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable security metrics."""

        return {
            "authentication_failures": (self.authentication_failures),
            "authorization_failures": (self.authorization_failures),
            "failure_counts": dict(self.failure_counts),
            "total_security_failures": (self.total_security_failures),
        }


class SecurityMetrics:
    """In-memory authentication and authorization metrics."""

    def __init__(self) -> None:
        self._authentication_failures = 0
        self._authorization_failures = 0
        self._failure_counts: dict[str, int] = {}

    def record_authentication_failure(
        self,
        *,
        failure_code: str,
    ) -> None:
        """Record one authentication failure."""

        self._authentication_failures += 1

        self._record_failure_code(failure_code)

    def record_authorization_failure(
        self,
        *,
        failure_code: str,
    ) -> None:
        """Record one authorization failure."""

        self._authorization_failures += 1

        self._record_failure_code(failure_code)

    def _record_failure_code(
        self,
        failure_code: str,
    ) -> None:
        self._failure_counts[failure_code] = (
            self._failure_counts.get(
                failure_code,
                0,
            )
            + 1
        )

    def snapshot(
        self,
    ) -> SecurityMetricsSnapshot:
        """Return a point-in-time security metrics snapshot."""

        return SecurityMetricsSnapshot(
            authentication_failures=(self._authentication_failures),
            authorization_failures=(self._authorization_failures),
            failure_counts=dict(self._failure_counts),
        )
