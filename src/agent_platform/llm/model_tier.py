from enum import IntEnum


class ModelCostTier(IntEnum):
    """Relative model cost tier used for routing constraints."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class ModelLatencyTier(IntEnum):
    """Relative model latency tier used for routing constraints."""

    FAST = 1
    STANDARD = 2
    SLOW = 3
