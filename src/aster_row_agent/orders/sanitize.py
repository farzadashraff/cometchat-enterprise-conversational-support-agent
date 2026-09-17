"""The order sanitizer: the one place `RawOrderRecord` is converted to `CustomerSafeOrder`.

This is an allowlist projection, not a denylist filter. `sanitize_order`
never writes `raw.customer` or `raw.internal` into anything — it is not
that those fields are read and then stripped out; the function simply
never references them at all, and `CustomerSafeOrder` has no field that
could hold their contents even if it tried (see `models.py`'s docstring).
That is what makes this a structural boundary rather than a convention:
adding `email=raw.customer.email` to the constructor call below would be
a `pydantic.ValidationError` at runtime (`CustomerSafeOrder` uses
`extra="forbid"`), not a silent leak.

Per the assignment's explicit precedence rule, `status` is authoritative
and is never inferred from other fields. Two deterministic rules live
here, in application code, precisely so a future LLM never has to get
them right on its own:

1. **Stale-field suppression** (INVARIANT 5): when `status` is
   `cancelled` or `returned`, `carrier`/`tracking_number`/
   `estimated_delivery` are withheld — an operational system may retain
   these after the fact, and surfacing them would imply the order is
   still in transit when it is not.
2. **No fabricated ETA** (INVARIANT 6, 7): `estimated_delivery` is
   passed through as-is when present and not suppressed; when the raw
   value is `None`, the safe value stays `None` — there is no code path
   in this function that computes, guesses, or defaults a date.

This module never infers policy, calls external systems, or modifies
source data — it is a pure function of its input.
"""

from __future__ import annotations

from aster_row_agent.orders.models import CustomerSafeOrder, CustomerSafeOrderItem, RawOrderRecord

# Per data/orders-data-dictionary.md: "Operational systems may retain stale
# carrier, tracking, or estimated-delivery fields after an order is
# cancelled or returned." These are the only two statuses this applies to —
# expressed as data, not as a per-order-ID special case.
_STALE_FIELD_STATUSES = frozenset({"cancelled", "returned"})

# Per data/orders-data-dictionary.md: "When status is exception, explain
# that support review is required and recommend a human handoff." This
# module does not perform the handoff (out of scope for this phase — see
# docs/architecture.md's handoff.py boundary); it only computes the
# deterministic flag a later phase's handoff logic will read.
_SUPPORT_REVIEW_STATUS = "exception"


def sanitize_order(raw: RawOrderRecord) -> CustomerSafeOrder:
    """Project a `RawOrderRecord` down to the customer-safe allowlist."""
    suppress_stale_fields = raw.status in _STALE_FIELD_STATUSES

    return CustomerSafeOrder(
        order_id=raw.order_id,
        membership_tier=raw.membership_tier,
        items=tuple(
            CustomerSafeOrderItem(
                name=item.name, quantity=item.quantity, final_sale=item.final_sale
            )
            for item in raw.items
        ),
        placed_at=raw.placed_at,
        status=raw.status,
        status_updated_at=raw.status_updated_at,
        shipped_at=raw.shipped_at,
        delivered_at=raw.delivered_at,
        carrier=None if suppress_stale_fields else raw.carrier,
        tracking_number=None if suppress_stale_fields else raw.tracking_number,
        estimated_delivery=None if suppress_stale_fields else raw.estimated_delivery,
        customer_safe_message=raw.customer_safe_message,
        requires_support_review=(raw.status == _SUPPORT_REVIEW_STATUS),
        stale_delivery_fields_suppressed=suppress_stale_fields,
    )
