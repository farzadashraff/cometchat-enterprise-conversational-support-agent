"""Heading-aware chunking.

Naive fixed-character splitting would sever a policy number from the
sentence that gives it meaning (e.g. splitting "30 calendar days" from
"of delivery"). Instead, the corpus's own Markdown heading structure is
used as the chunk boundary: one chunk per heading section by default, with
size-based sub-splitting only when a section is unusually large — and even
then, sub-splitting happens at paragraph boundaries, never mid-sentence,
and every resulting chunk keeps the full heading path of its section.

Every document in the supplied corpus is small enough (a handful of short
paragraphs per section) that no real section actually exceeds
``max_chunk_chars`` — see ``tests/unit/test_chunking.py`` for a synthetic
fixture that exercises the sub-splitting path directly, since the real
corpus cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aster_row_agent.rag.hashing import compute_chunk_id, hash_text
from aster_row_agent.rag.models import Document, DocumentChunk, HeadingPath

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class HeadingSection:
    """One contiguous span of body text attributed to a heading path.

    An intermediate parsing artifact (not part of the persisted domain
    model) produced by :func:`parse_heading_sections` and consumed by
    :func:`build_document_chunks`.
    """

    heading_path: tuple[str, ...]
    text: str
    order_index: int


def parse_heading_sections(body: str) -> list[HeadingSection]:
    """Walk a Markdown body and attribute each span of text to its heading path.

    Uses a simple stack of (level, heading_text): encountering a heading of
    level L pops every open heading of level >= L before pushing the new
    one, which correctly attributes nested subheadings (e.g. an H3 under an
    H2) without assuming any fixed document depth. Text appearing before
    the first heading (none of the 14 supplied documents have any) is still
    captured, with an empty heading path, rather than silently dropped.
    """
    stack: list[tuple[int, str]] = []
    sections: list[HeadingSection] = []
    buffer: list[str] = []
    order_index = 0

    def flush() -> None:
        nonlocal order_index
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            sections.append(
                HeadingSection(
                    heading_path=tuple(heading for _, heading in stack),
                    text=text,
                    order_index=order_index,
                )
            )
            order_index += 1

    for line in body.split("\n"):
        match = _HEADING_RE.match(line)
        if match is None:
            buffer.append(line)
            continue
        flush()
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading_text))
    flush()

    return sections


def split_section_text(text: str, max_chars: int) -> list[str]:
    """Split one section's text into pieces no larger than `max_chars`.

    Splits are made at paragraph boundaries (blank lines), packing
    paragraphs greedily so consecutive short paragraphs stay together.
    A single paragraph that alone exceeds `max_chars` is hard-split as a
    last resort (documented fallback; not expected to trigger on prose
    corpora with normal paragraph lengths).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            for start in range(0, len(paragraph), max_chars):
                pieces.append(paragraph[start : start + max_chars])

    if current:
        pieces.append(current)
    return pieces


def build_document_chunks(document: Document, max_chunk_chars: int) -> list[DocumentChunk]:
    """Chunk one parsed document into ordered, provenance-carrying chunks.

    `chunk_index` is a single monotonic counter across the whole document
    (not reset per section), so it reflects each chunk's true reading-order
    position and feeds directly into the deterministic chunk ID.
    """
    sections = parse_heading_sections(document.body)
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section in sections:
        for piece in split_section_text(section.text, max_chunk_chars):
            content_hash = hash_text(piece)
            chunk_id = compute_chunk_id(
                document_id=document.metadata.document_id,
                source_path=document.source_path,
                heading_path=section.heading_path,
                chunk_index=chunk_index,
                content_hash=content_hash,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.metadata.document_id,
                    source_path=document.source_path,
                    title=document.metadata.title,
                    heading_path=HeadingPath(parts=section.heading_path),
                    chunk_index=chunk_index,
                    text=piece,
                    content_hash=content_hash,
                    metadata=document.metadata,
                )
            )
            chunk_index += 1

    return chunks
