"""Exceptions raised by the order subsystem.

Follows the same pattern as `rag/errors.py`: every failure is a specific,
named subclass of :class:`OrderError` with enough context to diagnose it,
never a bare ``except Exception`` swallowing the real cause.
"""

from __future__ import annotations

from pathlib import Path


class OrderError(Exception):
    """Base class for all order-subsystem failures."""


class OrderDatasetNotFoundError(OrderError):
    """The configured orders file does not exist."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Orders dataset not found: {path}")


class OrderDatasetParseError(OrderError):
    """The orders file is not valid JSON, or its top-level shape is wrong."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse orders dataset {path}: {reason}")


class OrderRecordValidationError(OrderError):
    """One order record failed schema validation."""

    def __init__(self, index: int, order_id: str | None, cause: Exception) -> None:
        self.index = index
        self.order_id = order_id
        self.cause = cause
        label = order_id or f"at index {index}"
        super().__init__(f"Invalid order record ({label}): {cause}")


class OrderDatasetValidationError(OrderError):
    """One or more order records failed validation.

    Raised once, after attempting every record, so a single load reports
    every problem instead of stopping at the first one (mirrors
    `rag.errors.CorpusValidationError`).
    """

    def __init__(self, errors: list[OrderRecordValidationError]) -> None:
        self.errors = errors
        summary = "; ".join(str(error) for error in errors)
        super().__init__(f"{len(errors)} order record(s) failed validation: {summary}")


class DuplicateOrderIdError(OrderError):
    """Two or more order records share the same order_id.

    The repository must never silently pick one — a duplicate ID in a
    dataset that is supposed to key orders uniquely is a data-integrity
    failure, not a resolvable ambiguity.
    """

    def __init__(self, duplicate_ids: list[str]) -> None:
        self.duplicate_ids = duplicate_ids
        super().__init__(f"Duplicate order_id(s) in dataset: {sorted(set(duplicate_ids))}")


class OrderLookupInfrastructureError(OrderError):
    """Raised by an `OrderRepository` implementation when a lookup cannot be
    completed due to an infrastructure problem — never for "order not found."

    The in-memory, JSON-backed repository in this project never raises
    this in practice (a dict lookup cannot fail once constructed); the
    type exists so that `OrderLookupService`'s `dataset_error` outcome is
    a real, tested contract for repository implementations added later
    (e.g. a networked or database-backed one), not a hypothetical enum
    value nothing can ever trigger. See
    `tests/unit/test_order_service.py` for a fake repository that
    exercises this path.
    """
