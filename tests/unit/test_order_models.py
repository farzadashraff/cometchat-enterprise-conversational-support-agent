"""Property-style tests asserting the allowlist invariant itself, not just
today's known-forbidden fields.

The point of `test_customer_safe_order_field_set_is_exactly_the_documented_allowlist`
and `test_customer_safe_order_item_field_set_is_exactly_the_documented_allowlist`
is to make CI fail the moment someone adds a field to `CustomerSafeOrder`
without updating this test — turning "did we mean to widen the allowlist?"
into a question a reviewer is forced to answer, rather than something that
could slip through unnoticed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from aster_row_agent.orders.models import (
    CustomerSafeOrder,
    CustomerSafeOrderItem,
    RawCustomerInfo,
    RawInternalInfo,
    RawOrderItem,
    RawOrderRecord,
)
from aster_row_agent.orders.sanitize import sanitize_order

# The exact set of fields ever permitted on CustomerSafeOrder: the
# documented customer-safe fields from data/orders-data-dictionary.md,
# plus the two clearly-named derived/computed fields (see models.py).
_ALLOWED_ORDER_FIELDS = frozenset(
    {
        "order_id",
        "membership_tier",
        "items",
        "placed_at",
        "status",
        "status_updated_at",
        "shipped_at",
        "delivered_at",
        "carrier",
        "tracking_number",
        "estimated_delivery",
        "customer_safe_message",
        "requires_support_review",
        "stale_delivery_fields_suppressed",
    }
)

_ALLOWED_ITEM_FIELDS = frozenset({"name", "quantity", "final_sale"})

_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "customer",
        "name",  # as a top-level order field (customer name) — "name" IS
        # allowed on items, but must never appear as a top-level
        # CustomerSafeOrder field.
        "email",
        "shipping_address",
        "internal",
        "risk_score",
        "warehouse_note",
        "support_tags",
        "sku",
    }
)


def test_customer_safe_order_field_set_is_exactly_the_documented_allowlist() -> None:
    assert set(CustomerSafeOrder.model_fields.keys()) == _ALLOWED_ORDER_FIELDS


def test_customer_safe_order_item_field_set_is_exactly_the_documented_allowlist() -> None:
    assert set(CustomerSafeOrderItem.model_fields.keys()) == _ALLOWED_ITEM_FIELDS


def test_customer_safe_order_fields_are_a_subset_of_forbidden_complement() -> None:
    """CustomerSafeOrder fields ⊆ explicitly allowed field set, and
    disjoint from every field name known to carry sensitive data — checked
    generically, not by re-enumerating today's known-forbidden fields
    against today's known-safe fields (which would be circular)."""
    order_fields = set(CustomerSafeOrder.model_fields.keys())
    assert order_fields <= _ALLOWED_ORDER_FIELDS
    assert order_fields.isdisjoint(_FORBIDDEN_FIELD_NAMES - {"name"})  # "name" excluded: see above
    assert "customer" not in order_fields
    assert "internal" not in order_fields


def test_customer_safe_order_item_has_no_sku_field() -> None:
    assert "sku" not in CustomerSafeOrderItem.model_fields


def test_raw_order_record_top_level_fields_not_all_customer_safe() -> None:
    """Sanity check the two schemas are genuinely different sizes/shapes —
    guards against someone accidentally aliasing CustomerSafeOrder to
    RawOrderRecord."""
    raw_fields = set(RawOrderRecord.model_fields.keys())
    safe_fields = set(CustomerSafeOrder.model_fields.keys())
    assert "customer" in raw_fields
    assert "internal" in raw_fields
    assert "customer" not in safe_fields
    assert "internal" not in safe_fields


def test_property_sanitize_output_field_values_never_include_raw_only_data() -> None:
    """A randomized-ish property check across many synthetic raw records:
    for any combination of status/fields, the sanitized JSON never
    contains the raw customer/internal payload's content."""
    import itertools

    statuses = [
        "pending",
        "processing",
        "shipped",
        "delayed",
        "delivered",
        "returned",
        "cancelled",
        "exception",
    ]
    etas = [None, date(2026, 8, 22)]
    carriers = [None, "UPS"]

    for status, eta, carrier in itertools.product(statuses, etas, carriers):
        secret_email = f"secret-{status}-{eta}-{carrier}@example.test"
        secret_note = f"SECRET NOTE {status} {eta} {carrier}"
        raw = RawOrderRecord(
            order_id="ORD-1000",
            customer=RawCustomerInfo(
                name="Secret Name", email=secret_email, shipping_address="Secret Address 123"
            ),
            membership_tier="standard",
            items=(RawOrderItem(sku="SECRET-SKU", name="Item", quantity=1, final_sale=False),),
            placed_at=datetime(2026, 8, 1, tzinfo=UTC),
            status=status,
            status_updated_at=datetime(2026, 8, 2, tzinfo=UTC),
            shipped_at=None,
            delivered_at=None,
            carrier=carrier,
            tracking_number="TRACK-SECRET-1" if carrier else None,
            estimated_delivery=eta,
            customer_safe_message="Safe message.",
            internal=RawInternalInfo(
                risk_score=999, warehouse_note=secret_note, support_tags=("secret-tag",)
            ),
        )
        safe = sanitize_order(raw)
        dump = safe.model_dump_json()
        assert secret_email not in dump
        assert secret_note not in dump
        assert "Secret Name" not in dump
        assert "Secret Address" not in dump
        assert "SECRET-SKU" not in dump
        assert "999" not in dump
        assert "secret-tag" not in dump
