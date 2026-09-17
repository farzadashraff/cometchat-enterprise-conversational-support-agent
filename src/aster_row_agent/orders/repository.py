"""The order repository: loads, validates, and indexes `orders.json` once.

Storage, lookup, sanitization, and business/status logic are deliberately
kept in separate modules (`repository.py`, this file; `sanitize.py`;
`service.py` for orchestration) — this module's only job is "load the
dataset safely and answer `get_by_order_id` lookups." It never generates
a customer-facing response, never calls an LLM, and never performs RAG
retrieval.

Loading is fail-fast and happens once (per the assignment's small,
static dataset — no database, no reread-per-query): `OrderRepository.load`
either returns a fully validated repository or raises a specific
`OrderError` subclass. A dataset that fails to load is not something a
caller can partially work around; there is no supported "best effort"
mode.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pydantic

from aster_row_agent.orders.errors import (
    DuplicateOrderIdError,
    OrderDatasetNotFoundError,
    OrderDatasetParseError,
    OrderDatasetValidationError,
    OrderRecordValidationError,
)
from aster_row_agent.orders.models import RawOrderRecord


class OrderRepository:
    """An in-memory, load-once index of validated `RawOrderRecord`s, keyed by
    canonical `order_id`.

    Construct via `OrderRepository.load(path)`, never directly — the
    constructor takes already-validated data so that "an `OrderRepository`
    exists" is itself a guarantee the dataset was valid, not something
    each caller must separately remember to check.
    """

    def __init__(
        self, orders: dict[str, RawOrderRecord], *, snapshot_at: datetime, source_path: Path
    ) -> None:
        self._orders = orders
        self.snapshot_at = snapshot_at
        self.source_path = source_path

    def __len__(self) -> int:
        return len(self._orders)

    def get_by_order_id(self, canonical_order_id: str) -> RawOrderRecord | None:
        """Look up a single order by its canonical (`ORD-####`) ID.

        `canonical_order_id` is expected to already be normalized (see
        `orders/normalize.py`) — this method does not itself tolerate case
        or whitespace variance, keeping that concern in one place.
        """
        return self._orders.get(canonical_order_id)

    @classmethod
    def load(cls, path: Path) -> OrderRepository:
        """Load, parse, and validate the orders dataset from `path`.

        Raises a specific `OrderError` subclass for every distinct failure
        mode (missing file, invalid JSON, invalid records, duplicate IDs)
        — never a bare exception, and never a silently-repaired dataset.
        """
        if not path.is_file():
            raise OrderDatasetNotFoundError(path)

        raw_text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise OrderDatasetParseError(path, f"invalid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise OrderDatasetParseError(path, "top-level JSON value must be an object")

        snapshot_at_raw = payload.get("snapshot_at")
        if not isinstance(snapshot_at_raw, str):
            raise OrderDatasetParseError(path, "missing or invalid top-level 'snapshot_at'")
        try:
            snapshot_at = datetime.fromisoformat(snapshot_at_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise OrderDatasetParseError(path, f"invalid 'snapshot_at' timestamp: {exc}") from exc

        raw_orders = payload.get("orders")
        if not isinstance(raw_orders, list):
            raise OrderDatasetParseError(path, "missing or invalid top-level 'orders' array")

        records, validation_errors = _validate_records(raw_orders)
        if validation_errors:
            raise OrderDatasetValidationError(validation_errors)

        orders_by_id, duplicate_ids = _index_by_order_id(records)
        if duplicate_ids:
            raise DuplicateOrderIdError(duplicate_ids)

        return cls(orders_by_id, snapshot_at=snapshot_at, source_path=path)


def _validate_records(
    raw_orders: list[object],
) -> tuple[list[RawOrderRecord], list[OrderRecordValidationError]]:
    records: list[RawOrderRecord] = []
    errors: list[OrderRecordValidationError] = []
    for index, raw_order in enumerate(raw_orders):
        order_id = raw_order.get("order_id") if isinstance(raw_order, dict) else None
        try:
            records.append(RawOrderRecord.model_validate(raw_order))
        except pydantic.ValidationError as exc:
            errors.append(OrderRecordValidationError(index, order_id, exc))
    return records, errors


def _index_by_order_id(
    records: list[RawOrderRecord],
) -> tuple[dict[str, RawOrderRecord], list[str]]:
    orders_by_id: dict[str, RawOrderRecord] = {}
    duplicates: list[str] = []
    for record in records:
        if record.order_id in orders_by_id:
            duplicates.append(record.order_id)
            continue
        orders_by_id[record.order_id] = record
    return orders_by_id, duplicates
