from __future__ import annotations

import pytest
from pydantic import ValidationError

from aster_row_agent.rag.models import DocumentMetadata, HeadingPath


def test_document_metadata_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata.model_validate({"title": "Missing document_id"})


def test_document_metadata_rejects_blank_required_field() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata.model_validate(
            {
                "document_id": "   ",
                "title": "Doc",
                "status": "active",
                "audience": "customer",
                "policy_authority": "official",
            }
        )


def test_document_metadata_tolerates_missing_optional_fields() -> None:
    metadata = DocumentMetadata.model_validate(
        {
            "document_id": "DOC-1",
            "title": "Minimal Doc",
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
        }
    )
    assert metadata.effective_date is None
    assert metadata.last_reviewed is None
    assert metadata.supersedes is None
    assert metadata.superseded_by is None
    assert metadata.superseded_date is None
    assert metadata.customer_answering is None


def test_document_metadata_parses_dates() -> None:
    metadata = DocumentMetadata.model_validate(
        {
            "document_id": "DOC-1",
            "title": "Doc",
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
            "effective_date": "2026-04-01",
            "last_reviewed": "2026-07-15",
        }
    )
    assert metadata.effective_date is not None
    assert metadata.effective_date.isoformat() == "2026-04-01"
    assert metadata.last_reviewed is not None
    assert metadata.last_reviewed.isoformat() == "2026-07-15"


def test_document_metadata_preserves_unknown_extra_fields() -> None:
    """A future front-matter field this model doesn't yet know about must survive."""
    metadata = DocumentMetadata.model_validate(
        {
            "document_id": "DOC-1",
            "title": "Doc",
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
            "review_owner": "content-team",
        }
    )
    assert metadata.model_extra is not None
    assert metadata.model_extra.get("review_owner") == "content-team"
    # And it round-trips through serialization rather than being dropped.
    assert metadata.model_dump().get("review_owner") == "content-team"


def test_document_metadata_supports_customer_answering_flag() -> None:
    metadata = DocumentMetadata.model_validate(
        {
            "document_id": "MIG-TEST-04",
            "title": "Content Migration Scratchpad",
            "status": "draft",
            "audience": "internal",
            "policy_authority": "none",
            "customer_answering": False,
        }
    )
    assert metadata.customer_answering is False


def test_document_metadata_is_immutable() -> None:
    metadata = DocumentMetadata.model_validate(
        {
            "document_id": "DOC-1",
            "title": "Doc",
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
        }
    )
    with pytest.raises(ValidationError):
        metadata.status = "superseded"


def test_heading_path_str_format_matches_expected_citation_style() -> None:
    path = HeadingPath(parts=("Returns Policy", "Standard return window"))
    assert str(path) == "Returns Policy > Standard return window"


def test_heading_path_leaf() -> None:
    assert HeadingPath(parts=("A", "B", "C")).leaf == "C"
    assert HeadingPath(parts=()).leaf is None
