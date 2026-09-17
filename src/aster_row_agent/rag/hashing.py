"""Deterministic content hashing used for change detection and chunk identity.

Timestamps are never used as a proxy for "did the content change" — only
the actual bytes are hashed, so re-ingesting byte-identical source files
always yields identical hashes (and therefore identical chunk IDs), which
is what makes ingestion idempotent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def hash_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """Return the hex SHA-256 digest of UTF-8 encoded text."""
    return hash_bytes(text.encode("utf-8"))


# Truncating a SHA-256 digest to 24 hex chars (96 bits) keeps chunk IDs short
# and readable while remaining effectively collision-free at the corpus
# scale this project operates at (tens to low hundreds of chunks).
_CHUNK_ID_LENGTH = 24


def compute_chunk_id(
    *,
    document_id: str,
    source_path: str,
    heading_path: Sequence[str],
    chunk_index: int,
    content_hash: str,
) -> str:
    """Derive a stable, deterministic chunk ID from stable source properties.

    The ID is a pure function of (document_id, source_path, heading_path,
    chunk_index, content_hash) — never a random UUID — so re-ingesting an
    unchanged document reproduces byte-identical chunk IDs, and a change to
    a chunk's own text changes only that chunk's ID rather than every
    chunk's ID in the document.
    """
    canonical = "||".join(
        [
            document_id,
            source_path,
            ">".join(heading_path),
            str(chunk_index),
            content_hash,
        ]
    )
    return hash_text(canonical)[:_CHUNK_ID_LENGTH]
