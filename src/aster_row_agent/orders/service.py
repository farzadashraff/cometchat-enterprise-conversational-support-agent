"""The order lookup service: the tool-facing contract for order status.

    lookup_order(order_id) -> OrderLookupResult

`OrderLookupService.lookup` **is** the tool contract referenced in
docs/architecture.md — a future LLM tool-calling adapter (Phase 5) calls
this method directly and serializes its typed, already-sanitized
`OrderLookupResult` for the model. There is no supported path from this
service back to a `RawOrderRecord`: it orchestrates
`normalize_order_id` -> `OrderRepository.get_by_order_id` ->
`sanitize_order`, and the raw record never leaves the last of those three
calls.

This service does not answer policy questions, decide conversational
context, or generate natural-language responses (INVARIANT 13) — it
returns one typed result and nothing else. What a future agent *says*
about that result is entirely a later phase's concern.
"""

from __future__ import annotations

import logging
import time

from aster_row_agent.logging_setup import log_event
from aster_row_agent.orders.errors import OrderLookupInfrastructureError
from aster_row_agent.orders.models import CustomerSafeOrder, OrderLookupOutcome, OrderLookupResult
from aster_row_agent.orders.normalize import OrderIdNormalizationOutcome, normalize_order_id
from aster_row_agent.orders.repository import OrderRepository
from aster_row_agent.orders.sanitize import sanitize_order

logger = logging.getLogger(__name__)


class OrderLookupService:
    """Orchestrates normalization, repository lookup, and sanitization.

    Takes an already-loaded `OrderRepository` — there is no code path by
    which this service can be constructed with a dataset that failed to
    validate (see `OrderRepository.load`'s fail-fast contract).
    """

    def __init__(self, repository: OrderRepository) -> None:
        self._repository = repository

    def lookup(self, raw_order_id: str | None) -> OrderLookupResult:
        """Look up one order by (possibly messy) user-supplied input.

        Returns a fully typed `OrderLookupResult` — never raises for an
        ordinary "not found"/"malformed"/"missing" case, and never returns
        a `RawOrderRecord` under any outcome.
        """
        start = time.monotonic()
        input_length = _safe_length(raw_order_id)
        log_event(logger, logging.INFO, "order.lookup.started", requested_input_length=input_length)

        normalization = normalize_order_id(raw_order_id)

        if normalization.outcome is OrderIdNormalizationOutcome.MISSING:
            result = OrderLookupResult(
                outcome=OrderLookupOutcome.MISSING_ORDER_ID,
                requested_input=raw_order_id,
                canonical_order_id=None,
            )
            self._log_completed(result, start)
            return result

        if normalization.outcome is OrderIdNormalizationOutcome.MALFORMED:
            result = OrderLookupResult(
                outcome=OrderLookupOutcome.MALFORMED_ORDER_ID,
                requested_input=raw_order_id,
                canonical_order_id=None,
            )
            self._log_completed(result, start)
            return result

        canonical_order_id = normalization.canonical_order_id
        assert canonical_order_id is not None  # guaranteed by WELL_FORMED

        try:
            raw_record = self._repository.get_by_order_id(canonical_order_id)
        except OrderLookupInfrastructureError as exc:
            result = OrderLookupResult(
                outcome=OrderLookupOutcome.DATASET_ERROR,
                requested_input=raw_order_id,
                canonical_order_id=canonical_order_id,
                error_category=type(exc).__name__,
            )
            self._log_completed(result, start)
            return result

        if raw_record is None:
            result = OrderLookupResult(
                outcome=OrderLookupOutcome.NOT_FOUND,
                requested_input=raw_order_id,
                canonical_order_id=canonical_order_id,
            )
            self._log_completed(result, start)
            return result

        safe_order = sanitize_order(raw_record)
        result = OrderLookupResult(
            outcome=OrderLookupOutcome.FOUND,
            requested_input=raw_order_id,
            canonical_order_id=canonical_order_id,
            order=safe_order,
        )
        self._log_completed(result, start)
        return result

    def _log_completed(self, result: OrderLookupResult, start_time: float) -> None:
        duration_ms = (time.monotonic() - start_time) * 1000
        fields: dict[str, object] = {
            "canonical_order_id": result.canonical_order_id,
            "outcome": result.outcome.value,
            "duration_ms": round(duration_ms, 3),
        }
        if result.order is not None:
            fields.update(_safe_order_log_fields(result.order))
        if result.error_category is not None:
            fields["error_category"] = result.error_category
        log_event(logger, logging.INFO, "order.lookup.completed", **fields)


def _safe_order_log_fields(order: CustomerSafeOrder) -> dict[str, object]:
    """Fields safe to log for a found order — every one of these is already
    on `CustomerSafeOrder`, so this function cannot accidentally reach
    into `RawOrderRecord`; there is nothing here to reach into."""
    return {
        "status": order.status,
        "eta_available": order.estimated_delivery is not None,
        "stale_delivery_fields_suppressed": order.stale_delivery_fields_suppressed,
        "requires_support_review": order.requires_support_review,
    }


def _safe_length(raw_order_id: str | None) -> int:
    return len(raw_order_id) if raw_order_id is not None else 0
