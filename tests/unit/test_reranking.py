import pytest
from pydantic import ValidationError

from agent_platform.memory import (
    DeterministicReranker,
    KnowledgeChunk,
    KnowledgeClassification,
    KnowledgeQuery,
    KnowledgeRankingService,
    KnowledgeRetrievalResult,
    KnowledgeSourceType,
    Reranker,
    RerankRequest,
    RetrievedKnowledgeChunk,
)


def create_candidate(
    *,
    chunk_id: str,
    content: str,
    score: float,
    rank: int,
) -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        chunk=KnowledgeChunk(
            chunk_id=chunk_id,
            document_id=f"{chunk_id}-document",
            tenant_id="tenant-001",
            content=content,
            chunk_index=0,
            source_type=KnowledgeSourceType.DOCUMENT,
            classification=KnowledgeClassification.INTERNAL,
        ),
        score=score,
        rank=rank,
    )


def create_query(
    *,
    top_k: int = 5,
) -> KnowledgeQuery:
    return KnowledgeQuery(
        query="incident escalation policy",
        tenant_id="tenant-001",
        top_k=top_k,
    )


def test_deterministic_reranker_satisfies_protocol() -> None:
    reranker = DeterministicReranker()

    assert isinstance(
        reranker,
        Reranker,
    )


def test_rerank_request_requires_positive_top_k() -> None:
    with pytest.raises(ValidationError):
        RerankRequest(
            query=create_query(),
            candidates=(),
            top_k=0,
        )


@pytest.mark.asyncio
async def test_exact_term_overlap_ranks_first() -> None:
    reranker = DeterministicReranker()

    request = RerankRequest(
        query=create_query(),
        candidates=(
            create_candidate(
                chunk_id="shipment",
                content="Shipment tracking procedure.",
                score=0.99,
                rank=1,
            ),
            create_candidate(
                chunk_id="incident",
                content=("Incident escalation policy for priority cases."),
                score=0.70,
                rank=2,
            ),
        ),
        top_k=2,
    )

    result = await reranker.rerank(
        request=request,
    )

    assert result.items[0].item.chunk.chunk_id == "incident"

    assert result.items[1].item.chunk.chunk_id == "shipment"


@pytest.mark.asyncio
async def test_original_vector_score_breaks_rerank_tie() -> None:
    reranker = DeterministicReranker()

    request = RerankRequest(
        query=create_query(),
        candidates=(
            create_candidate(
                chunk_id="lower",
                content="Incident operations manual.",
                score=0.60,
                rank=2,
            ),
            create_candidate(
                chunk_id="higher",
                content="Incident response guide.",
                score=0.90,
                rank=1,
            ),
        ),
        top_k=2,
    )

    result = await reranker.rerank(
        request=request,
    )

    assert result.items[0].item.chunk.chunk_id == "higher"


@pytest.mark.asyncio
async def test_reranking_assigns_new_sequential_ranks() -> None:
    reranker = DeterministicReranker()

    result = await reranker.rerank(
        request=RerankRequest(
            query=create_query(),
            candidates=(
                create_candidate(
                    chunk_id="one",
                    content="Incident.",
                    score=0.5,
                    rank=7,
                ),
                create_candidate(
                    chunk_id="two",
                    content="Incident escalation policy.",
                    score=0.4,
                    rank=8,
                ),
            ),
            top_k=2,
        )
    )

    assert tuple(item.rank for item in result.items) == (
        1,
        2,
    )


@pytest.mark.asyncio
async def test_reranking_respects_top_k() -> None:
    reranker = DeterministicReranker()

    candidates = tuple(
        create_candidate(
            chunk_id=f"chunk-{index}",
            content=f"Incident escalation policy {index}",
            score=float(index) / 10,
            rank=index + 1,
        )
        for index in range(5)
    )

    result = await reranker.rerank(
        request=RerankRequest(
            query=create_query(),
            candidates=candidates,
            top_k=2,
        )
    )

    assert len(result.items) == 2


@pytest.mark.asyncio
async def test_ranking_service_returns_canonical_retrieval_result() -> None:
    service = KnowledgeRankingService(
        reranker=DeterministicReranker(),
    )

    original = KnowledgeRetrievalResult(
        query=create_query(
            top_k=2,
        ),
        items=(
            create_candidate(
                chunk_id="unrelated",
                content="Shipment tracking.",
                score=0.95,
                rank=1,
            ),
            create_candidate(
                chunk_id="relevant",
                content="Incident escalation policy.",
                score=0.80,
                rank=2,
            ),
        ),
    )

    result = await service.rank(
        result=original,
    )

    assert isinstance(
        result,
        KnowledgeRetrievalResult,
    )

    assert result.items[0].chunk.chunk_id == "relevant"


@pytest.mark.asyncio
async def test_empty_retrieval_result_is_returned_unchanged() -> None:
    service = KnowledgeRankingService(
        reranker=DeterministicReranker(),
    )

    original = KnowledgeRetrievalResult(
        query=create_query(),
        items=(),
    )

    result = await service.rank(
        result=original,
    )

    assert result == original
