"""Exceptions raised by the ingestion pipeline.

All are subclasses of :class:`IngestionError` so callers (the CLI, tests)
can catch one type and still distinguish specific failures via `isinstance`
or the message. Ingestion never swallows a failure behind a bare
``except Exception`` — every failure surfaces as one of these, with enough
context (source path, underlying cause) to diagnose it directly from the
error message and structured logs.
"""

from __future__ import annotations

from pathlib import Path


class IngestionError(Exception):
    """Base class for all ingestion failures."""


class CorpusNotFoundError(IngestionError):
    """The configured knowledge-base directory does not exist or is not a directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Knowledge-base directory not found or not a directory: {path}")


class UnsafeCorpusPathError(IngestionError):
    """A discovered path resolves outside the configured knowledge-base root.

    Guards against a malicious or accidental symlink inside the corpus
    directory pointing at an arbitrary filesystem location.
    """

    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        self.root = root
        super().__init__(f"Refusing to ingest path outside corpus root: {path} (root={root})")


class FrontMatterError(IngestionError):
    """The document's YAML front matter is missing or malformed."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Invalid front matter in {path}: {reason}")


class DocumentParsingError(IngestionError):
    """A single document failed to parse (front matter or metadata validation)."""

    def __init__(self, path: Path, cause: Exception) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"Failed to parse {path}: {cause}")


class CorpusValidationError(IngestionError):
    """One or more documents in the corpus failed to parse.

    Raised once, after attempting every document, so a single ingestion run
    reports every problem instead of stopping at the first one.
    """

    def __init__(self, errors: list[DocumentParsingError]) -> None:
        self.errors = errors
        summary = "; ".join(str(error) for error in errors)
        super().__init__(f"{len(errors)} document(s) failed to parse: {summary}")


class IndexNotFoundError(IngestionError):
    """An attempt was made to load an index that has not been built yet."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"No index found at {path}. Run ingestion first.")
