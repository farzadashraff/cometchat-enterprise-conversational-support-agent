from __future__ import annotations

from aster_row_agent.rag.chunking import build_document_chunks
from aster_row_agent.rag.claims import extract_claims
from aster_row_agent.rag.models import Document, DocumentChunk, DocumentMetadata


def _metadata(**overrides: object) -> DocumentMetadata:
    base: dict[str, object] = {
        "document_id": "DOC-1",
        "title": "Doc",
        "status": "active",
        "audience": "customer",
        "policy_authority": "official",
    }
    base.update(overrides)
    return DocumentMetadata.model_validate(base)


def _single_chunk(body: str, **metadata_overrides: object) -> DocumentChunk:
    document = Document(
        source_path="knowledge-base/doc.md",
        metadata=_metadata(**metadata_overrides),
        body=body,
        content_hash="irrelevant",
    )
    chunks = build_document_chunks(document, max_chunk_chars=4000)
    assert len(chunks) == 1
    return chunks[0]


# --- return_window_days --------------------------------------------------------


def test_extracts_plain_calendar_days_return_window() -> None:
    chunk = _single_chunk(
        "# Returns Policy\n\n## Standard return window\n\n"
        "Customers may request a return within 30 calendar days of delivery.\n"
    )
    claims = [c for c in extract_claims(chunk) if c.concept == "return_window_days"]
    assert len(claims) == 1
    assert claims[0].value == 30


def test_extracts_hyphenated_calendar_day_form() -> None:
    chunk = _single_chunk(
        "# TrailPlus\n\n## Return window\n\n"
        "A member receives a 45-calendar-day return window from delivery.\n"
    )
    claims = [c for c in extract_claims(chunk) if c.concept == "return_window_days"]
    assert len(claims) == 1
    assert claims[0].value == 45


def test_extracts_plain_days_without_calendar_word() -> None:
    chunk = _single_chunk(
        "# Scratchpad\n\n## Draft\n\n"
        "Every customer receives 60 days to return every item.\n"
    )
    claims = [c for c in extract_claims(chunk) if c.concept == "return_window_days"]
    assert len(claims) == 1
    assert claims[0].value == 60


def test_does_not_extract_refund_timing_business_days() -> None:
    """'5-7 business days' is refund *processing* time, not a return window,
    and the sentence stating it does not mention "return" — must not be
    conflated with the concept."""
    chunk = _single_chunk(
        "# Returns\n\n## Refunds\n\n"
        "Refunds are issued after the return is inspected. "
        "Customers should allow 5-7 business days after inspection for the refund to appear.\n"
    )
    claims = [c for c in extract_claims(chunk) if c.concept == "return_window_days"]
    assert claims == []


def test_does_not_extract_unrelated_reporting_window() -> None:
    """A damaged-item reporting window ('report ... within 7 calendar days')
    is a different concept and must not be extracted as a return window."""
    chunk = _single_chunk(
        "# Damaged Items\n\n## Reporting window\n\n"
        "Customers should report an item that arrived damaged within 7 calendar days of delivery.\n"
    )
    claims = [c for c in extract_claims(chunk) if c.concept == "return_window_days"]
    assert claims == []


def test_return_window_claim_carries_applicability_tags_from_local_sentence() -> None:
    chunk = _single_chunk(
        "# Returns\n\n## Standard return window\n\n"
        "Customers on the standard plan may request a return within 30 calendar days of delivery. "
        "TrailPlus members receive a different return window.\n"
    )
    claims = [c for c in extract_claims(chunk) if c.concept == "return_window_days"]
    assert len(claims) == 1
    assert "standard" in claims[0].applicability_tags
    assert "trailplus" not in claims[0].applicability_tags


# --- breeze_tumbler_body_dishwasher_safe ---------------------------------------


def test_extracts_all_components_dishwasher_safe_as_true() -> None:
    chunk = _single_chunk(
        "# Breeze Tumbler — Product Information\n\n## Cleaning\n\n"
        "The product card states that all components are dishwasher safe.\n"
    )
    claims = [
        c for c in extract_claims(chunk) if c.concept == "breeze_tumbler_body_dishwasher_safe"
    ]
    assert len(claims) == 1
    assert claims[0].value is True


def test_extracts_hand_washed_as_false() -> None:
    chunk = _single_chunk(
        "# Product Care Guide\n\n## Breeze Tumbler\n\n"
        "The stainless-steel body of the Breeze Tumbler should be hand-washed.\n"
    )
    claims = [
        c for c in extract_claims(chunk) if c.concept == "breeze_tumbler_body_dishwasher_safe"
    ]
    assert len(claims) == 1
    assert claims[0].value is False


def test_gated_on_tumbler_word_present_somewhere_in_chunk() -> None:
    """A hand-wash instruction for an unrelated product must not be mistaken
    for a claim about the tumbler."""
    chunk = _single_chunk(
        "# Product Care Guide\n\n## Packing cubes\n\n"
        "Packing cubes may be hand-washed in cool water.\n"
    )
    claims = [
        c for c in extract_claims(chunk) if c.concept == "breeze_tumbler_body_dishwasher_safe"
    ]
    assert claims == []


def test_tumble_dry_does_not_false_positive_as_tumbler_word() -> None:
    """'tumble dry' must not match the \\btumbler\\b gate."""
    chunk = _single_chunk(
        "# Product Care Guide\n\n## Bags and backpacks\n\n"
        "Do not machine wash, bleach, dry-clean, or tumble dry. The item should be hand-washed.\n"
    )
    claims = [
        c for c in extract_claims(chunk) if c.concept == "breeze_tumbler_body_dishwasher_safe"
    ]
    assert claims == []


def test_extract_claims_aggregates_across_multiple_concepts() -> None:
    """A chunk could in principle trigger more than one extractor; extract_claims
    must not stop at the first match."""
    chunk = _single_chunk(
        "# Breeze Tumbler — Product Information\n\n## Cleaning\n\n"
        "You may return this tumbler within 30 calendar days. "
        "The product card states that all components are dishwasher safe.\n"
    )
    concepts = {c.concept for c in extract_claims(chunk)}
    assert concepts == {"return_window_days", "breeze_tumbler_body_dishwasher_safe"}
