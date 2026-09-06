from pydantic import BaseModel, ConfigDict, Field


class RetrievalEvaluationCase(BaseModel):
    """One deterministic retrieval-quality evaluation scenario."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    case_id: str = Field(
        min_length=1,
    )

    query: str = Field(
        min_length=1,
    )

    expected_relevant_chunk_ids: frozenset[str]

    evaluation_k: int = Field(
        default=5,
        ge=1,
    )


class RetrievalEvaluationMetrics(BaseModel):
    """Quality metrics for one retrieval evaluation case."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    hit_rate_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    recall_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    precision_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    reciprocal_rank: float = Field(
        ge=0.0,
        le=1.0,
    )


class RetrievalEvaluationResult(BaseModel):
    """Evaluation result for one retrieval case."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    case: RetrievalEvaluationCase

    retrieved_chunk_ids: tuple[str, ...]

    relevant_retrieved_chunk_ids: tuple[str, ...]

    metrics: RetrievalEvaluationMetrics

    passed: bool


class RetrievalEvaluationSummary(BaseModel):
    """Aggregated results across multiple evaluation cases."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    total_cases: int = Field(
        ge=0,
    )

    passed_cases: int = Field(
        ge=0,
    )

    failed_cases: int = Field(
        ge=0,
    )

    mean_hit_rate_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_recall_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_precision_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_reciprocal_rank: float = Field(
        ge=0.0,
        le=1.0,
    )
