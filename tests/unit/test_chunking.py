from __future__ import annotations

from aster_row_agent.rag.chunking import (
    build_document_chunks,
    parse_heading_sections,
    split_section_text,
)
from aster_row_agent.rag.models import Document, DocumentMetadata


def _metadata(**overrides: object) -> DocumentMetadata:
    base: dict[str, object] = {
        "document_id": "DOC-1",
        "title": "Test Document",
        "status": "active",
        "audience": "customer",
        "policy_authority": "official",
    }
    base.update(overrides)
    return DocumentMetadata.model_validate(base)


def _document(body: str, **metadata_overrides: object) -> Document:
    return Document(
        source_path="knowledge-base/test-doc.md",
        metadata=_metadata(**metadata_overrides),
        body=body,
        content_hash="irrelevant-for-chunking-tests",
    )


# --- parse_heading_sections -------------------------------------------------


def test_parse_heading_sections_basic_two_level() -> None:
    body = "# Returns Policy\n\n## Standard return window\n\nCustomers may return within 30 days.\n"
    sections = parse_heading_sections(body)
    assert len(sections) == 1
    assert sections[0].heading_path == ("Returns Policy", "Standard return window")
    assert sections[0].text == "Customers may return within 30 days."


def test_parse_heading_sections_multiple_sibling_headings() -> None:
    body = (
        "# Returns Policy\n\n"
        "## Standard return window\n\nFirst section text.\n\n"
        "## Item condition\n\nSecond section text.\n"
    )
    sections = parse_heading_sections(body)
    assert [s.heading_path for s in sections] == [
        ("Returns Policy", "Standard return window"),
        ("Returns Policy", "Item condition"),
    ]
    assert sections[0].text == "First section text."
    assert sections[1].text == "Second section text."


def test_parse_heading_sections_preserves_order_index() -> None:
    body = "# Title\n\n## A\n\ntext a\n\n## B\n\ntext b\n"
    sections = parse_heading_sections(body)
    assert [s.order_index for s in sections] == [0, 1]


def test_parse_heading_sections_handles_nested_subheadings() -> None:
    body = (
        "# Title\n\n"
        "## Section\n\nIntro under section.\n\n"
        "### Subsection\n\nText under subsection.\n"
    )
    sections = parse_heading_sections(body)
    assert sections[0].heading_path == ("Title", "Section")
    assert sections[0].text == "Intro under section."
    assert sections[1].heading_path == ("Title", "Section", "Subsection")
    assert sections[1].text == "Text under subsection."


def test_parse_heading_sections_preamble_before_first_heading_is_not_dropped() -> None:
    body = "Some preamble text with no heading above it.\n\n# Title\n\n## Section\n\nBody.\n"
    sections = parse_heading_sections(body)
    assert sections[0].heading_path == ()
    assert "preamble" in sections[0].text
    assert sections[1].heading_path == ("Title", "Section")


def test_parse_heading_sections_ignores_headings_with_no_body_text() -> None:
    body = "# Title\n\n## Empty Section\n\n## Section With Text\n\nContent here.\n"
    sections = parse_heading_sections(body)
    assert len(sections) == 1
    assert sections[0].heading_path == ("Title", "Section With Text")


def test_parse_heading_sections_returns_empty_list_for_blank_document() -> None:
    assert parse_heading_sections("") == []
    assert parse_heading_sections("\n\n   \n") == []


# --- split_section_text -----------------------------------------------------


def test_split_section_text_returns_single_piece_when_under_limit() -> None:
    text = "Short section text."
    assert split_section_text(text, max_chars=1200) == [text]


def test_split_section_text_returns_empty_list_for_blank_text() -> None:
    assert split_section_text("   \n  ", max_chars=1200) == []


def test_split_section_text_packs_paragraphs_greedily() -> None:
    paragraphs = ["Paragraph one.", "Paragraph two.", "Paragraph three."]
    text = "\n\n".join(paragraphs)
    # max_chars big enough for two short paragraphs together but not all three
    max_chars = len(paragraphs[0]) + len(paragraphs[1]) + 2
    pieces = split_section_text(text, max_chars=max_chars)
    assert len(pieces) == 2
    assert "Paragraph one." in pieces[0]
    assert "Paragraph two." in pieces[0]
    assert pieces[1] == "Paragraph three."


def test_split_section_text_never_exceeds_max_chars_per_piece() -> None:
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(20))
    pieces = split_section_text(text, max_chars=100)
    assert all(len(piece) <= 100 for piece in pieces)


def test_split_section_text_hard_splits_single_oversized_paragraph() -> None:
    huge_paragraph = "word " * 500  # no blank lines at all: one giant paragraph
    pieces = split_section_text(huge_paragraph, max_chars=200)
    assert len(pieces) > 1
    assert all(len(piece) <= 200 for piece in pieces)
    # Reassembling the pieces reproduces the original text losslessly.
    assert "".join(pieces) == huge_paragraph.strip()


# --- build_document_chunks ---------------------------------------------------


def test_build_document_chunks_matches_documented_example_shape() -> None:
    """Mirrors the exact example in docs/architecture.md and the Phase 2 brief."""
    document = _document(
        "# Returns Policy\n\n## Standard return window\n\nCustomers on the standard plan...\n",
        document_id="RET-2026-01",
        title="Returns Policy",
    )
    chunks = build_document_chunks(document, max_chunk_chars=1200)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "RET-2026-01"
    assert chunk.heading_path.parts == ("Returns Policy", "Standard return window")
    assert chunk.text == "Customers on the standard plan..."
    assert chunk.citation == "test-doc.md — Returns Policy > Standard return window"


def test_build_document_chunks_does_not_split_small_sections() -> None:
    document = _document(
        "# Title\n\n## A\n\nShort text A.\n\n## B\n\nShort text B.\n"
    )
    chunks = build_document_chunks(document, max_chunk_chars=1200)
    assert len(chunks) == 2  # one chunk per heading, not merged, not fragmented


def test_build_document_chunks_splits_oversized_section_preserving_heading_path() -> None:
    paragraphs = "\n\n".join(f"Filler paragraph {i} with several words in it." for i in range(30))
    document = _document(f"# Title\n\n## Big Section\n\n{paragraphs}\n")
    chunks = build_document_chunks(document, max_chunk_chars=150)
    assert len(chunks) > 1
    assert all(chunk.heading_path.parts == ("Title", "Big Section") for chunk in chunks)
    # chunk_index must be strictly increasing and contiguous from zero
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_build_document_chunks_ids_are_deterministic_across_calls() -> None:
    document = _document(
        "# Title\n\n## A\n\nText A.\n\n## B\n\nText B.\n"
    )
    chunks_first = build_document_chunks(document, max_chunk_chars=1200)
    chunks_second = build_document_chunks(document, max_chunk_chars=1200)
    assert [c.chunk_id for c in chunks_first] == [c.chunk_id for c in chunks_second]


def test_build_document_chunks_ids_are_unique_within_a_document() -> None:
    document = _document(
        "# Title\n\n## A\n\nText A.\n\n## B\n\nText B.\n\n## C\n\nText C.\n"
    )
    chunks = build_document_chunks(document, max_chunk_chars=1200)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_build_document_chunks_carries_full_metadata_onto_every_chunk() -> None:
    document = _document(
        "# Title\n\n## A\n\nText A.\n",
        status="superseded",
        audience="internal",
        policy_authority="none",
    )
    chunks = build_document_chunks(document, max_chunk_chars=1200)
    assert chunks[0].metadata.status == "superseded"
    assert chunks[0].metadata.audience == "internal"
    assert chunks[0].metadata.policy_authority == "none"
