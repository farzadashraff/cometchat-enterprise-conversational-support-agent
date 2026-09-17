from __future__ import annotations

from aster_row_agent.rag.applicability import applicability_tags


def test_detects_trailplus() -> None:
    assert "trailplus" in applicability_tags("A TrailPlus member receives 45 days.")


def test_detects_standard_plan() -> None:
    assert "standard" in applicability_tags("Customers on the standard plan get 30 days.")


def test_detects_final_sale_hyphenated_and_spaced() -> None:
    assert "final_sale" in applicability_tags("This is a final-sale item.")
    assert "final_sale" in applicability_tags("This is a final sale item.")


def test_detects_domestic() -> None:
    assert "domestic" in applicability_tags("Domestic orders ship in 3-5 days.")


def test_detects_international() -> None:
    assert "international" in applicability_tags("International shipping is limited.")


def test_detects_canada_and_canadian() -> None:
    assert "canada" in applicability_tags("We ship to Canada.")
    assert "canada" in applicability_tags("Canadian returns are handled differently.")


def test_unscoped_text_yields_empty_tags() -> None:
    assert applicability_tags("Refunds are issued within a few business days.") == frozenset()


def test_can_detect_multiple_tags_in_one_sentence() -> None:
    tags = applicability_tags("TrailPlus members get free domestic shipping.")
    assert {"trailplus", "domestic"} <= tags


def test_universal_scope_overrides_incidental_segment_mention() -> None:
    """'every item, including ... final-sale merchandise' is a claim about
    *everything*, not one narrowly scoped to final-sale items — see the
    real migration-scratchpad sentence this is modeled on."""
    text = "Every customer receives 60 days to return every item, including final-sale merchandise."
    assert applicability_tags(text) == frozenset()


def test_universal_scope_all_customers_phrase() -> None:
    assert applicability_tags("All customers get free shipping on TrailPlus orders.") == frozenset()


def test_is_case_insensitive() -> None:
    assert "canada" in applicability_tags("shipping to CANADA")
    assert "trailplus" in applicability_tags("trailplus membership")
