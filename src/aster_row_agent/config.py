"""Central configuration for the project.

Every module that needs a path, model name, or tunable threshold reads it
from :class:`Settings` rather than hardcoding it or reading the environment
directly. Values may be overridden via environment variables (prefixed
``ASTER_ROW_``) or a local ``.env`` file — see ``.env.example``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for ingestion (and, in later phases, the agent)."""

    model_config = SettingsConfigDict(
        env_prefix="ASTER_ROW_",
        env_file=".env",
        extra="ignore",
    )

    knowledge_base_dir: Path = Path("knowledge-base")
    """Directory containing the supplied, immutable Markdown corpus."""

    orders_file: Path = Path("data/orders.json")
    """The supplied, immutable mock order dataset. Not to be confused with
    `data_dir` below (note: no leading dot) — this is supplied input,
    that is generated output."""

    evaluation_dir: Path = Path("evaluation")
    """Directory containing `visible-cases.json` (supplied verbatim) and
    `custom-cases.json` (original Phase 7 cases) — see
    `evaluation/runner.py`."""

    data_dir: Path = Path(".data")
    """Root directory for generated, gitignored artifacts (index, caches)."""

    index_dir_override: Path | None = None
    """Explicit index directory, bypassing `data_dir`. Set via `--index-dir`."""

    embedding_cache_dir_override: Path | None = None
    """Explicit embedding cache directory, bypassing `data_dir`."""

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    """Identifier of the local embedding model used to build the vector index."""

    max_chunk_chars: int = 1200
    """Soft upper bound (characters) for a single chunk before safe sub-splitting."""

    log_level: str = "INFO"

    llm_model: str = "claude-haiku-4-5-20251001"
    """The model name passed to the LLM provider. The API key itself is
    read directly from the `ANTHROPIC_API_KEY` environment variable by
    the `anthropic` SDK (never through this settings object, and never
    logged) — see `agent/llm.py::AnthropicLLMClient`."""

    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 30.0

    @property
    def index_dir(self) -> Path:
        """Where the derived, rebuildable RAG index is written."""
        return self.index_dir_override or (self.data_dir / "index")

    @property
    def embedding_cache_dir(self) -> Path:
        """Where the local embedding model's downloaded weights are cached."""
        return self.embedding_cache_dir_override or (self.data_dir / "embedding_cache")


def get_settings() -> Settings:
    """Load settings from the environment/`.env` file.

    A thin function (rather than a module-level singleton) so tests can call
    this fresh with monkeypatched environment variables without import-order
    surprises.
    """
    return Settings()
