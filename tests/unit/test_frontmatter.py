from __future__ import annotations

from pathlib import Path

import pytest

from aster_row_agent.rag.errors import FrontMatterError
from aster_row_agent.rag.frontmatter import split_front_matter

DUMMY_PATH = Path("dummy.md")


def test_split_front_matter_parses_well_formed_document() -> None:
    raw = (
        "---\ndocument_id: DOC-1\ntitle: Test Doc\nstatus: active\n"
        "---\n# Test Doc\n\nBody text.\n"
    )
    front_matter, body = split_front_matter(raw, path=DUMMY_PATH)
    assert front_matter == {"document_id": "DOC-1", "title": "Test Doc", "status": "active"}
    assert body.strip() == "# Test Doc\n\nBody text."


def test_split_front_matter_preserves_body_exactly() -> None:
    raw = "---\ndocument_id: DOC-1\n---\n# Heading\n\nLine one.\nLine two.\n"
    _, body = split_front_matter(raw, path=DUMMY_PATH)
    assert body == "# Heading\n\nLine one.\nLine two.\n"


def test_split_front_matter_does_not_mutate_input_string() -> None:
    raw = "---\ndocument_id: DOC-1\n---\nBody\n"
    original = raw
    split_front_matter(raw, path=DUMMY_PATH)
    assert raw == original


def test_split_front_matter_missing_opening_delimiter_raises() -> None:
    raw = "# No front matter here\n\nJust body text.\n"
    with pytest.raises(FrontMatterError):
        split_front_matter(raw, path=DUMMY_PATH)


def test_split_front_matter_missing_closing_delimiter_raises() -> None:
    raw = "---\ndocument_id: DOC-1\n\n# Body without a closing delimiter\n"
    with pytest.raises(FrontMatterError):
        split_front_matter(raw, path=DUMMY_PATH)


def test_split_front_matter_invalid_yaml_raises() -> None:
    raw = "---\ndocument_id: [unterminated\n---\nBody\n"
    with pytest.raises(FrontMatterError):
        split_front_matter(raw, path=DUMMY_PATH)


def test_split_front_matter_non_mapping_yaml_raises() -> None:
    raw = "---\n- item_one\n- item_two\n---\nBody\n"
    with pytest.raises(FrontMatterError):
        split_front_matter(raw, path=DUMMY_PATH)


def test_split_front_matter_empty_front_matter_yields_empty_dict() -> None:
    raw = "---\n---\nBody\n"
    front_matter, body = split_front_matter(raw, path=DUMMY_PATH)
    assert front_matter == {}
    assert body.strip() == "Body"


def test_split_front_matter_error_includes_path_for_diagnosis() -> None:
    raw = "no front matter\n"
    with pytest.raises(FrontMatterError) as exc_info:
        split_front_matter(raw, path=Path("knowledge-base/broken.md"))
    assert "broken.md" in str(exc_info.value)
