"""Typed domain models for the order boundary.

Two families of models live here, and the split between them is the
single most important thing in this package:

- `Raw*` models mirror the supplied `orders.json` schema exactly,
  including the fields that must never reach an LLM (`RawCustomerInfo`,
  `RawInternalInfo`). These are trusted **application** data — never
  LLM-facing.
- `CustomerSafeOrder` (and `CustomerSafeOrderItem`) is a *separate*,
  narrower model containing only fields explicitly permitted by
  `data/orders-data-dictionary.md`. It has no field that could hold an
  email, address, or internal note — not "a field that is emptied out,"
  a field that structurally does not exist on the type. A future
  agent/tool layer only ever sees this second family.

`Raw*` models use `extra="allow"`: an order record that gains a new
field in the future must not crash the whole lookup subsystem (matches
the `DocumentMetadata` precedent in `rag/models.py`). This is safe
specifically *because* `CustomerSafeOrder` is built by an explicit
allowlist in `sanitize.py` that only ever reads known field names off
the raw model — a new raw field is captured (so it isn't silently lost)
but is never read by the sanitizer, so it can never become customer-safe
by accident. `CustomerSafeOrder` itself uses `extra="forbid"`: it is the
strict boundary type, and nothing should ever be able to attach an
unexpected field to it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RawCustomerInfo(BaseModel):
    """Customer PII. Never read outside `orders/repository.py` internals —
    no other module in this package, and no module outside it, should ever
    import or access this type's fields."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str
    email: str
    shipping_address: str


class RawOrderItem(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    sku: str
    name: str
    quantity: int
    final_sale: bool


class RawInternalInfo(BaseModel):
    """Internal-only operational data. Never read outside `orders/repository.py`
    internals. Note this may contain adversarial text (see
    docs/corpus-analysis.md and docs/security-model.md) — it is untrusted
    data even within the trusted application boundary, and is never
    interpreted as instructions anywhere in this codebase."""

    model_config = ConfigDict(extra="allow", frozen=True)

    risk_score: int
    warehouse_note: str
    support_tags: tuple[str, ...]


class RawOrderRecord(BaseModel):
    """One order exactly as stored in `orders.json`. Trusted application
    data — but NEVER an LLM-facing type (see this module's docstring and
    INVARIANT 1 in docs/security-model.md)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    order_id: str
    customer: RawCustomerInfo
    membership_tier: str
    items: tuple[RawOrderItem, ...]
    placed_at: datetime
    status: str
    status_updated_at: datetime
    shipped_at: datetime | None
    delivered_at: datetime | None
    carrier: str | None
    tracking_number: str | None
    estimated_delivery: date | None
    customer_safe_message: str
    internal: RawInternalInfo


class CustomerSafeOrderItem(BaseModel):
    """Item fields explicitly permitted by the data dictionary — `sku` is
    deliberately excluded; it is not in the documented customer-safe list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    quantity: int
    final_sale: bool


class CustomerSafeOrder(BaseModel):
    """The complete allowlist of fields ever permitted to reach a customer
    or an LLM prompt for an order-status answer.

    Every field here corresponds to an entry in `data/orders-data-dictionary.md`'s
    "Customer-safe fields" list, except the two clearly-named derived
    fields at the end, which are computed *from* already-safe fields
    (never from anything in `RawCustomerInfo`/`RawInternalInfo`) so a
    future agent doesn't need to re-derive business rules from a raw
    status string:

    - `requires_support_review`: `True` iff `status == "exception"`.
    - `stale_delivery_fields_suppressed`: `True` iff `carrier`/
      `tracking_number`/`estimated_delivery` were withheld because
      `status` is `cancelled` or `returned` (see `sanitize.py`) — lets a
      consumer distinguish "this order never had a carrier" from "this
      order had one, but it's stale and was withheld."
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str
    membership_tier: str
    items: tuple[CustomerSafeOrderItem, ...]
    placed_at: datetime
    status: str
    status_updated_at: datetime
    shipped_at: datetime | None
    delivered_at: datetime | None
    carrier: str | None
    tracking_number: str | None
    estimated_delivery: date | None
    customer_safe_message: str
    requires_support_review: bool
    stale_delivery_fields_suppressed: bool


class OrderLookupOutcome(StrEnum):
    """Every state a lookup attempt can end in. A future agent must branch
    on this, never infer the outcome from whether a field happens to be
    `None`."""

    FOUND = "found"
    MISSING_ORDER_ID = "missing_order_id"
    """No order ID was supplied at all — the agent should ask for one."""
    MALFORMED_ORDER_ID = "malformed_order_id"
    """Something was supplied but it is not a plausible order ID — the
    agent should ask for a valid one, distinct from asking for one at all."""
    NOT_FOUND = "not_found"
    """A well-formed ID was supplied but no such order exists."""
    DATASET_ERROR = "dataset_error"
    """The lookup could not be completed due to an infrastructure problem.
    Must never be presented as "order not found" — that would be
    factually wrong (INVARIANT 10)."""


class OrderLookupResult(BaseModel):
    """The complete, typed result of one lookup attempt — the only thing
    `OrderLookupService.lookup()` ever returns. `order` is populated if
    and only if `outcome is OrderLookupOutcome.FOUND`."""

    model_config = ConfigDict(frozen=True)

    outcome: OrderLookupOutcome
    requested_input: str | None
    """The raw input as supplied, kept only for trace/debug — this is
    always just an attempted order-id-shaped string, never customer data."""
    canonical_order_id: str | None
    """The normalized `ORD-####` form, when normalization succeeded
    (populated for FOUND and NOT_FOUND; None for MISSING/MALFORMED/error)."""
    order: CustomerSafeOrder | None = None
    error_category: str | None = None
    """A short, safe category label for DATASET_ERROR (e.g. an exception
    class name) — never a raw exception message that might embed
    internal details."""
