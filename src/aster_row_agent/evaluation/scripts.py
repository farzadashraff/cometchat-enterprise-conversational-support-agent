"""Per-case scripted LLM answers for cases that reach the generation step.

Per the Phase 7 brief §23, the core evaluation suite runs without a real
Anthropic API key — every case uses `FakeLLMClient`. For cases whose
turn is fully resolved by deterministic short-circuits (a sensitive
request, a missing/malformed order ID, an unknown order, an evidence-
insufficient question with no order found, ...), `Agent` never calls the
LLM at all, so no script is needed or provided; an empty list is used
deliberately so any *unexpected* LLM call raises loudly inside the
harness, turning "the code called the LLM when it should have
short-circuited" into a directly visible evaluation failure.

For every other case, an answer is authored by hand from the real,
supplied knowledge-base text — this tests the *rest* of the pipeline
(citation resolution against the real evidence bundle, response
validation, handoff computation, rendering) against a plausible, well-
behaved model output. It is not a test of live model phrasing quality;
see docs/evaluation-plan.md's "two-tier design" note for the honest
limitation this implies.

Each entry is a list of `(answer_text, cited_filenames)` pairs, one per
LLM call expected across the case's whole session (in order).
"""

from __future__ import annotations

SCRIPTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    # --- visible cases -----------------------------------------------------------------
    "standard-return-window": [
        (
            "Customers on the standard plan may request a return within 30 calendar "
            "days of delivery.",
            ("01-returns-policy-current.md",),
        )
    ],
    "valid-order-lookup": [
        (
            "Your order ORD-1007 has shipped via UPS and is estimated to arrive August 22, 2026.",
            (),
        )
    ],
    "trailplus-return-window": [
        (
            "As a TrailPlus member, you have a return window of 45 calendar days from delivery.",
            ("09-trailplus-membership.md",),
        )
    ],
    "final-sale-damaged-exception": [
        (
            "Final sale does not block a damaged-item review — please report it "
            "within 7 days of delivery so our team can review it and confirm next "
            "steps.",
            ("03-final-sale-and-promotions.md", "04-damaged-or-wrong-items.md"),
        )
    ],
    "canada-multiturn": [
        ("Yes, Aster & Row ships internationally to Canada.", ("06-international-shipping.md",)),
        (
            "Yes, Canada is a supported destination. Canadian orders generally "
            "arrive within 5-9 business days after dispatch, and duties or taxes "
            "are not prepaid by us — the recipient is responsible for those "
            "charges.",
            ("06-international-shipping.md",),
        ),
    ],
    "unsupported-country": [
        (
            "Shipping to Germany is not currently available — we currently ship "
            "internationally only to Canada.",
            ("06-international-shipping.md",),
        )
    ],
    "no-lifetime-warranty": [
        (
            "No, there is no lifetime warranty. Bags have 2 years of warranty "
            "coverage, and drinkware and travel accessories have 1 year.",
            ("07-warranty.md",),
        )
    ],
    "retrieved-prompt-injection": [
        (
            "That migration note is not authoritative — our standard policy is 30 "
            "days unless a valid exception applies, and the agent cannot approve a "
            "return directly; a human team member handles approvals.",
            ("01-returns-policy-current.md",),
        )
    ],
    "insufficient-information": [
        (
            "I don't have confirmation that all fabrics and adhesives used in our "
            "bags are vegan. I'd recommend checking with our support team for more "
            "details.",
            (),
        )
    ],
    "genuine-active-source-conflict": [
        (
            "Our current official sources conflict on this: one says to hand-wash "
            "the tumbler body, and another says all components are dishwasher "
            "safe. I'd recommend hand-washing until we can confirm which is "
            "correct with a human team member.",
            ("11-product-care.md", "12-breeze-tumbler-product-card.md"),
        )
    ],
    "cancelled-order-stale-eta": [
        (
            "This order is cancelled, so it will not be shipped.",
            (),
        )
    ],
    "shipped-without-eta": [
        (
            "Your order shipped with Canada Post. A delivery estimate is "
            "unavailable for this shipment right now.",
            (),
        )
    ],
    # unknown-order, missing-order-id, order-data-privacy: deterministic
    # short-circuits, no LLM call — no script.
    # --- custom cases --------------------------------------------------------------
    "paraphrased-policy-question": [
        (
            "You're welcome to return most items within 30 calendar days of "
            "delivery under our standard return window.",
            ("01-returns-policy-current.md",),
        )
    ],
    "order-plus-policy-combo": [
        (
            "I can't process a refund directly, but ORD-1001 can be reviewed "
            "under our damaged/late item policy — a human team member can "
            "confirm the resolution.",
            ("04-damaged-or-wrong-items.md",),
        )
    ],
    "unsupported-action-cancel-now": [
        (
            "I'm not able to cancel an order directly. ORD-1001 is still within "
            "its eligibility window, and a human team member can process the "
            "cancellation for you.",
            (),
        )
    ],
    "malformed-id-combined-with-policy": [
        (
            "That order ID doesn't look valid, so I wasn't able to look it up — "
            "could you double-check it? In the meantime, our standard return "
            "window is 30 calendar days from delivery.",
            ("01-returns-policy-current.md",),
        )
    ],
    "superseded-policy-paraphrase": [
        (
            "Our current policy is a 30 calendar day return window; the 45-day "
            "figure was from an older, superseded policy and is no longer in "
            "effect.",
            ("01-returns-policy-current.md",),
        )
    ],
    "breeze-conflict-paraphrase": [
        (
            "Our current official sources conflict on this: one says to hand-wash "
            "the tumbler body, and another says all components are dishwasher "
            "safe. I'd recommend hand-washing until we can confirm with a human "
            "team member.",
            ("11-product-care.md", "12-breeze-tumbler-product-card.md"),
        )
    ],
    "order-id-normalization-lowercase-whitespace": [
        (
            "Your order ORD-1007 has shipped via UPS and is estimated to arrive August 22, 2026.",
            (),
        )
    ],
    # ambiguous-order-followup-no-id, system-prompt-extraction-paraphrase,
    # unsupported-vegan-attribute-standalone, session-isolation-order-context,
    # unrelated-topic-then-vegan-question (2nd turn only): deterministic
    # short-circuits — no script, or a script only for their one LLM-reaching turn.
    "unrelated-topic-then-vegan-question": [
        ("Yes, we ship internationally to Canada.", ("06-international-shipping.md",)),
    ],
    "session-isolation-order-context": [
        ("Your order ORD-1001 is currently pending and has not shipped yet.", ()),
    ],
    "short-unrelated-question-after-conflict-topic": [
        (
            "Our current official sources conflict on this: one says to hand-wash "
            "the tumbler body, and another says all components are dishwasher "
            "safe. I'd recommend hand-washing until we can confirm with a human "
            "team member.",
            ("11-product-care.md", "12-breeze-tumbler-product-card.md"),
        )
    ],
}
