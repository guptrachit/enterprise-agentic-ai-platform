from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReadinessStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class Layer3ReadinessCheck(BaseModel):
    """One production-readiness assertion."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    check_id: str = Field(
        min_length=1,
    )

    description: str = Field(
        min_length=1,
    )

    status: ReadinessStatus

    detail: str | None = None


class Layer3ReadinessReport(BaseModel):
    """Aggregated readiness status for Layer 3."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    checks: tuple[Layer3ReadinessCheck, ...]

    @property
    def passed(
        self,
    ) -> bool:
        return all(check.status == ReadinessStatus.PASS for check in self.checks)

    @property
    def failed_check_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            check.check_id
            for check in self.checks
            if check.status == ReadinessStatus.FAIL
        )
