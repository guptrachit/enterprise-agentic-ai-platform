from agent_platform.memory.reranking import (
    RerankedKnowledgeChunk,
    RerankRequest,
    RerankResult,
)


class DeterministicReranker:
    """Simple lexical reranker for deterministic tests."""

    async def rerank(
        self,
        *,
        request: RerankRequest,
    ) -> RerankResult:
        query_terms = self._tokenize(request.query.query)

        scored_items: list[
            tuple[
                object,
                float,
            ]
        ] = []

        for candidate in request.candidates:
            content_terms = self._tokenize(candidate.chunk.content)

            if not query_terms:
                score = 0.0
            else:
                overlap = len(query_terms.intersection(content_terms))

                score = overlap / len(query_terms)

            scored_items.append(
                (
                    candidate,
                    score,
                )
            )

        scored_items.sort(
            key=lambda item: (
                item[1],
                item[0].score if item[0].score is not None else 0.0,
            ),
            reverse=True,
        )

        items = tuple(
            RerankedKnowledgeChunk(
                item=candidate,
                rerank_score=score,
                rank=rank,
            )
            for rank, (
                candidate,
                score,
            ) in enumerate(
                scored_items[: request.top_k],
                start=1,
            )
        )

        return RerankResult(
            query=request.query,
            items=items,
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> frozenset[str]:
        normalized = "".join(
            character.casefold() if character.isalnum() else " " for character in text
        )

        return frozenset(token for token in normalized.split() if token)
