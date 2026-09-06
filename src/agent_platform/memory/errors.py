class MemoryError(Exception):
    """Base exception for governed memory behavior."""


class MemoryRecordNotFoundError(MemoryError):
    """Raised when a required memory record cannot be found."""


class MemoryStoreError(MemoryError):
    """Raised when the underlying memory store cannot complete an operation."""
