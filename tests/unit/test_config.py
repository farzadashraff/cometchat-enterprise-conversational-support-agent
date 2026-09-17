from __future__ import annotations

from pathlib import Path

import pytest

from aster_row_agent.config import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in list(__import__("os").environ):
        if var.startswith("ASTER_ROW_"):
            monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.knowledge_base_dir == Path("knowledge-base")
    assert settings.data_dir == Path(".data")
    assert settings.index_dir == Path(".data/index")
    assert settings.embedding_cache_dir == Path(".data/embedding_cache")
    assert settings.max_chunk_chars == 1200


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTER_ROW_MAX_CHUNK_CHARS", "500")
    monkeypatch.setenv("ASTER_ROW_EMBEDDING_MODEL_NAME", "some/other-model")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.max_chunk_chars == 500
    assert settings.embedding_model_name == "some/other-model"


def test_settings_index_dir_override_bypasses_data_dir() -> None:
    settings = Settings(_env_file=None, index_dir_override=Path("/tmp/custom-index"))  # type: ignore[call-arg]
    assert settings.index_dir == Path("/tmp/custom-index")
