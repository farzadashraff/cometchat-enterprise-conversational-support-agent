"""Order-ID normalization and validation.

Deliberately conservative, per the assignment's explicit instruction not
to "guess a substantially different order ID when the supplied value does
not match" (`data/orders-data-dictionary.md`). This module tolerates only
harmless formatting variance — case, surrounding whitespace, and the
separator between "ORD" and the digits — never fuzzy or partial matching.
`"1007"` is never turned into `"ORD-1007"`: nothing in the assignment
supports that guess, so it is treated as malformed instead.

Because the accepted pattern is an anchored allowlist (`^ORD` ... digits
... `$`), any input that doesn't look exactly like an order ID — a
filesystem path, a shell fragment, a SQL string, arbitrary Unicode, an
oversized string — simply fails to match and is classified
`MALFORMED`, with no special-casing required. See
`tests/unit/test_order_normalize.py` for adversarial-input coverage.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

# Real IDs in the supplied dataset are `ORD-1001`..`ORD-1012` (4 digits).
# The 4-10 digit range is a generous, still-safe upper bound: long enough
# that a legitimate future order number never gets rejected, short enough
# that a pathological numeric string ("ORD-" + one million digits) is
# still capped rather than accepted.
_ORDER_ID_PATTERN = re.compile(r"^ORD[\s\-_]*([0-9]{4,10})$", re.IGNORECASE)

# A hard ceiling checked before any regex work, purely defensive: no
# legitimate order ID is anywhere near this long, and rejecting early
# means an adversarial multi-kilobyte string is never logged, retried
# against a second pattern, or otherwise processed further.
_MAX_INPUT_LENGTH = 64


class OrderIdNormalizationOutcome(StrEnum):
    MISSING = "missing"
    """No input was supplied at all (`None` or all-whitespace)."""
    MALFORMED = "malformed"
    """Something was supplied but does not match a plausible order ID."""
    WELL_FORMED = "well_formed"
    """Matches the canonical `ORD-####` shape after harmless normalization."""


class OrderIdNormalization(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: OrderIdNormalizationOutcome
    canonical_order_id: str | None
    """Populated only when `outcome is WELL_FORMED`."""


def normalize_order_id(raw: str | None) -> OrderIdNormalization:
    """Classify and (if possible) canonicalize a supplied order-ID string.

    This function does not look up the ID against the dataset — it only
    decides whether the *shape* of the input is a plausible order ID.
    Whether that ID actually exists is `OrderRepository`'s concern.
    """
    if raw is None or not raw.strip():
        return OrderIdNormalization(
            outcome=OrderIdNormalizationOutcome.MISSING, canonical_order_id=None
        )

    stripped = raw.strip()
    if len(stripped) > _MAX_INPUT_LENGTH:
        return OrderIdNormalization(
            outcome=OrderIdNormalizationOutcome.MALFORMED, canonical_order_id=None
        )

    match = _ORDER_ID_PATTERN.match(stripped)
    if match is None:
        return OrderIdNormalization(
            outcome=OrderIdNormalizationOutcome.MALFORMED, canonical_order_id=None
        )

    digits = match.group(1)
    return OrderIdNormalization(
        outcome=OrderIdNormalizationOutcome.WELL_FORMED, canonical_order_id=f"ORD-{digits}"
    )
