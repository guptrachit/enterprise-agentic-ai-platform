from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class RoutingPolicyVersion:
    """Semantic version identifier for an LLM routing policy."""

    major: int
    minor: int
    patch: int = 0

    def __post_init__(self) -> None:
        """Validate semantic-version components."""

        if self.major < 0:
            raise ValueError("major version must be greater than or equal to 0")

        if self.minor < 0:
            raise ValueError("minor version must be greater than or equal to 0")

        if self.patch < 0:
            raise ValueError("patch version must be greater than or equal to 0")

    @property
    def value(self) -> str:
        """Return the semantic version string."""

        return f"{self.major}.{self.minor}.{self.patch}"

    def __str__(self) -> str:
        """Return the semantic version string."""

        return self.value

    @classmethod
    def parse(
        cls,
        value: str,
    ) -> "RoutingPolicyVersion":
        """Parse a semantic version string."""

        parts = value.strip().split(".")

        if len(parts) not in (
            2,
            3,
        ):
            raise ValueError(
                "routing policy version must use 'major.minor' or 'major.minor.patch'"
            )

        try:
            numbers = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError(
                "routing policy version components must be integers"
            ) from error

        if len(numbers) == 2:
            major, minor = numbers

            return cls(
                major=major,
                minor=minor,
            )

        major, minor, patch = numbers

        return cls(
            major=major,
            minor=minor,
            patch=patch,
        )
