from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkingMemoryItem(BaseModel):
    """Single temporary fact used during one governed agent execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    key: str = Field(min_length=1)
    value: Any
    source: str | None = None


class WorkingMemory(BaseModel):
    """Immutable temporary memory for a single agent execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    execution_id: str = Field(min_length=1)
    items: tuple[WorkingMemoryItem, ...] = ()

    def set(
        self,
        *,
        key: str,
        value: Any,
        source: str | None = None,
    ) -> "WorkingMemory":
        """Return a new working-memory object with the key inserted or replaced."""

        new_item = WorkingMemoryItem(
            key=key,
            value=value,
            source=source,
        )

        remaining_items = tuple(item for item in self.items if item.key != key)

        return self.model_copy(
            update={
                "items": (
                    *remaining_items,
                    new_item,
                )
            }
        )

    def get(
        self,
        key: str,
    ) -> Any | None:
        """Return the value associated with a key, if present."""

        for item in self.items:
            if item.key == key:
                return item.value

        return None

    def get_item(
        self,
        key: str,
    ) -> WorkingMemoryItem | None:
        """Return the full working-memory item, including provenance."""

        for item in self.items:
            if item.key == key:
                return item

        return None

    def remove(
        self,
        key: str,
    ) -> "WorkingMemory":
        """Return a new working-memory object without the requested key."""

        return self.model_copy(
            update={"items": tuple(item for item in self.items if item.key != key)}
        )

    def contains(
        self,
        key: str,
    ) -> bool:
        """Return whether a key exists in working memory."""

        return any(item.key == key for item in self.items)
