import pytest
from pydantic import ValidationError

from agent_platform.memory import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingService,
    EmbeddingVector,
)


def test_embedding_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(
            text="",
        )


def test_embedding_vector_requires_values() -> None:
    with pytest.raises(ValidationError):
        EmbeddingVector(
            values=(),
        )


def test_embedding_vector_reports_dimensions() -> None:
    vector = EmbeddingVector(
        values=(
            0.1,
            0.2,
            0.3,
        )
    )

    assert vector.dimensions == 3


def test_deterministic_provider_satisfies_protocol() -> None:
    provider = DeterministicEmbeddingProvider(
        dimensions=4,
    )

    assert isinstance(
        provider,
        EmbeddingProvider,
    )


def test_deterministic_provider_requires_positive_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        DeterministicEmbeddingProvider(
            dimensions=0,
        )


@pytest.mark.asyncio
async def test_same_text_produces_same_embedding() -> None:
    provider = DeterministicEmbeddingProvider(
        dimensions=4,
    )

    request = EmbeddingRequest(
        text="enterprise retrieval policy",
    )

    first = await provider.embed(
        request=request,
    )

    second = await provider.embed(
        request=request,
    )

    assert first.vector == second.vector


@pytest.mark.asyncio
async def test_embedding_has_expected_dimensions() -> None:
    provider = DeterministicEmbeddingProvider(
        dimensions=6,
    )

    result = await provider.embed(
        request=EmbeddingRequest(
            text="customer support policy",
        )
    )

    assert result.vector.dimensions == 6
    assert result.provider == "deterministic"
    assert result.model == "deterministic-6d"


@pytest.mark.asyncio
async def test_embedding_service_delegates_to_provider() -> None:
    service = EmbeddingService(
        provider=DeterministicEmbeddingProvider(
            dimensions=4,
        )
    )

    result = await service.embed_text(
        text="incident escalation",
    )

    assert result.vector.dimensions == 4


@pytest.mark.asyncio
async def test_embed_many_preserves_input_order() -> None:
    service = EmbeddingService(
        provider=DeterministicEmbeddingProvider(
            dimensions=4,
        )
    )

    texts = (
        "first document",
        "second document",
        "third document",
    )

    results = await service.embed_many(
        texts=texts,
    )

    expected = tuple(
        [
            await service.embed_text(
                text=text,
            )
            for text in texts
        ]
    )

    assert results == expected


@pytest.mark.asyncio
async def test_different_text_can_produce_different_vectors() -> None:
    provider = DeterministicEmbeddingProvider(
        dimensions=4,
    )

    first = await provider.embed(
        request=EmbeddingRequest(
            text="customer escalation policy",
        )
    )

    second = await provider.embed(
        request=EmbeddingRequest(
            text="pharmacy shipment status",
        )
    )

    assert first.vector != second.vector
