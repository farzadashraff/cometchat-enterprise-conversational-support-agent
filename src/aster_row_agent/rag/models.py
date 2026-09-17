"""Typed domain models for the RAG evidence layer.

These models are the canonical representation that flows through ingestion
and, later, retrieval — no unstructured dicts are passed between pipeline
stages. Everything here is pure data: no authority ranking, no conflict
resolution, no citation-selection logic. That business logic belongs to the
retrieval/authority layer added in a later phase (see docs/architecture.md
§4-5); mixing it in here would blur the ingestion/retrieval boundary the
Phase 1 architecture calls for.

A deliberate security property lives in how `DocumentMetadata` is modeled:
`status`, `audience`, and `policy_authority` are preserved as plain strings
exactly as authored, for *every* document — including the internal,
draft, and deliberately adversarial ones (see docs/corpus-analysis.md).
Nothing in this module drops, rewrites, or "cleans up" a document because
of what its metadata says; it only makes that metadata available, honestly
and completely, for a later layer to act on.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class DocumentMetadata(BaseModel):
    """Parsed YAML front matter for one knowledge-base document.

    Only the five fields observed on every document in the supplied corpus
    (`document_id`, `title`, `status`, `audience`, `policy_authority`) are
    required. Everything else is optional because the corpus does not
    guarantee it is present (e.g. `supersedes` only appears on the current
    returns policy; `customer_answering` only appears on the migration
    scratchpad) — a future document lacking one of these must not fail
    ingestion. Unrecognized front-matter keys are preserved rather than
    discarded (`extra="allow"`), so a new metadata field introduced later
    survives ingestion even before this model is updated to know about it.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    document_id: str
    title: str
    status: str
    audience: str
    policy_authority: str

    effective_date: date | None = None
    last_reviewed: date | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    superseded_date: date | None = None
    customer_answering: bool | None = None

    @field_validator("document_id", "title", "status", "audience", "policy_authority")
    @classmethod
    def _non_empty(cls, value: str, info: ValidationInfo) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return stripped


class Document(BaseModel):
    """One fully parsed source file, before chunking."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    """Path relative to the project root, e.g. 'knowledge-base/01-returns-policy-current.md'."""

    metadata: DocumentMetadata
    body: str
    """Markdown body with the front-matter block removed. Never written back to disk."""

    content_hash: str
    """SHA-256 of the raw file bytes (front matter + body), for change detection."""


class HeadingPath(BaseModel):
    """An ordered path of Markdown headings, e.g. ('Returns Policy', 'Standard return window')."""

    model_config = ConfigDict(frozen=True)

    parts: tuple[str, ...] = ()

    def __str__(self) -> str:
        return " > ".join(self.parts)

    @property
    def leaf(self) -> str | None:
        return self.parts[-1] if self.parts else None


class DocumentChunk(BaseModel):
    """One retrievable, heading-aware passage, carrying full provenance.

    A `DocumentChunk` is self-sufficient for citation: a consumer needs
    nothing beyond this object to produce a citation such as
    "01-returns-policy-current.md — Returns Policy > Standard return
    window" — provenance is carried as structured data, not left for an
    LLM to remember.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source_path: str
    title: str
    heading_path: HeadingPath
    chunk_index: int
    """0-based ordinal position of this chunk within its source document."""
    text: str
    content_hash: str
    """SHA-256 of `text` alone (distinct from the parent document's content_hash)."""
    metadata: DocumentMetadata

    @property
    def filename(self) -> str:
        return Path(self.source_path).name

    @property
    def citation(self) -> str:
        """Human-readable source reference: '<filename> — <heading path>'."""
        heading = str(self.heading_path)
        return f"{self.filename} — {heading}" if heading else self.filename


class IndexRecord(BaseModel):
    """One chunk plus its embedding, as persisted in the local index."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    embedding: tuple[float, ...]
    embedding_model: str


class SearchResult(BaseModel):
    """One ranked hit returned by a similarity search over the index."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float


class ChunkingConfig(BaseModel):
    """Records which chunking strategy/parameters produced a manifest's chunks."""

    model_config = ConfigDict(frozen=True)

    strategy: str = "heading-aware-v1"
    max_chunk_chars: int
    version: int = 1


class DocumentManifestEntry(BaseModel):
    """Per-document summary recorded in the corpus manifest."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    document_id: str
    title: str
    status: str
    audience: str
    policy_authority: str
    content_hash: str
    chunk_count: int


class CorpusManifest(BaseModel):
    """Machine-readable record of one ingestion run, for reproducibility and debugging.

    This is the artifact a reviewer (or CI) can inspect to answer "what
    corpus, in what state, produced this index?" without re-running
    ingestion or reading logs.
    """

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    corpus_fingerprint: str
    """Deterministic hash over every (source_path, content_hash) pair, sorted."""
    knowledge_base_dir: str
    document_count: int
    chunk_count: int
    documents: tuple[DocumentManifestEntry, ...]
    embedding_model: str
    embedding_dimensions: int
    chunking: ChunkingConfig
    index_schema_version: int = 1
    ingestion_duration_ms: float
