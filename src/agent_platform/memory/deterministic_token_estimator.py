class DeterministicTokenEstimator:
    """Simple deterministic estimator for architecture and tests."""

    def estimate(
        self,
        *,
        text: str,
    ) -> int:
        normalized = text.strip()

        if not normalized:
            return 1

        return max(
            1,
            len(normalized.split()),
        )
