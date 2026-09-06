from pydantic import BaseModel, ConfigDict, Field

from agent_platform.memory.retrieval_evaluation import (
    RetrievalEvaluationMetrics,
)


class RetrievalEvaluationThresholds(BaseModel):
    """Minimum acceptable retrieval quality."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    minimum_hit_rate_at_k: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    minimum_recall_at_k: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    minimum_precision_at_k: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    minimum_reciprocal_rank: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
    )


class RetrievalEvaluationPolicy:
    """Applies deterministic pass/fail thresholds."""

    def __init__(
        self,
        *,
        thresholds: RetrievalEvaluationThresholds,
    ) -> None:
        self._thresholds = thresholds

    def passes(
        self,
        *,
        metrics: RetrievalEvaluationMetrics,
    ) -> bool:
        return all(
            (
                metrics.hit_rate_at_k >= self._thresholds.minimum_hit_rate_at_k,
                metrics.recall_at_k >= self._thresholds.minimum_recall_at_k,
                metrics.precision_at_k >= self._thresholds.minimum_precision_at_k,
                metrics.reciprocal_rank >= self._thresholds.minimum_reciprocal_rank,
            )
        )
