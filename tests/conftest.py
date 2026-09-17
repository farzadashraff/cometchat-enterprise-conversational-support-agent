from __future__ import annotations

from pathlib import Path

import pytest

from aster_row_agent.config import Settings
from aster_row_agent.orders.repository import OrderRepository
from aster_row_agent.orders.service import OrderLookupService
from aster_row_agent.rag.embeddings import EmbeddingProvider, FastEmbedProvider
from aster_row_agent.rag.evidence import EvidenceAssembler
from aster_row_agent.rag.index import load_index_records
from aster_row_agent.rag.ingest import run_ingestion
from aster_row_agent.rag.models import IndexRecord
from aster_row_agent.rag.retrieval import Retriever
from aster_row_agent.rag.retrieval_models import EvidenceAssemblyOptions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"
REAL_ORDERS_FILE = PROJECT_ROOT / "data" / "orders.json"

# All 14 supplied documents, by filename, with the metadata this test suite
# treats as ground truth (taken directly from docs/corpus-analysis.md, not
# from evaluation/visible-cases.json — no visible-case wording or expected
# answers are encoded here).
EXPECTED_DOCUMENT_COUNT = 14

# The 12 supplied mock orders — ground truth taken directly from
# data/orders.json, not from evaluation/visible-cases.json.
EXPECTED_ORDER_COUNT = 12


@pytest.fixture
def real_knowledge_base_dir() -> Path:
    """The project's actual, immutable knowledge-base directory."""
    assert REAL_KNOWLEDGE_BASE_DIR.is_dir(), "knowledge-base/ must exist for these tests"
    return REAL_KNOWLEDGE_BASE_DIR


@pytest.fixture
def real_orders_file() -> Path:
    """The project's actual, immutable orders dataset."""
    assert REAL_ORDERS_FILE.is_file(), "data/orders.json must exist for these tests"
    return REAL_ORDERS_FILE


@pytest.fixture(scope="session")
def real_order_repository() -> OrderRepository:
    return OrderRepository.load(REAL_ORDERS_FILE)


@pytest.fixture
def real_order_lookup_service(real_order_repository: OrderRepository) -> OrderLookupService:
    return OrderLookupService(real_order_repository)


# --- Real-embedding fixtures for retrieval/evidence tests -------------------
#
# Retrieval quality (does a paraphrase actually rank the right document,
# does an off-topic query actually score low) cannot be meaningfully tested
# with a hash-based fake embedding, since it has no notion of semantic
# similarity at all — see rag/embeddings.py. These tests therefore use the
# real local embedding model (no API key, no network after the first
# cached download — see docs/architecture.md §Phase 2 as-built). Every
# fixture here is session-scoped so the model loads and the corpus is
# embedded exactly once per test run (see the Phase 3 brief §20:
# "embeddings are not recomputed for every query").


@pytest.fixture(scope="session")
def real_embedding_provider() -> EmbeddingProvider:
    settings = Settings()
    return FastEmbedProvider(settings.embedding_model_name, settings.embedding_cache_dir)


@pytest.fixture(scope="session")
def real_index_records(
    real_embedding_provider: EmbeddingProvider, tmp_path_factory: pytest.TempPathFactory
) -> list[IndexRecord]:
    data_dir = tmp_path_factory.mktemp("real-corpus-index")
    settings = Settings(knowledge_base_dir=REAL_KNOWLEDGE_BASE_DIR, data_dir=data_dir)
    run_ingestion(settings, real_embedding_provider)
    return load_index_records(settings.index_dir)


@pytest.fixture(scope="session")
def real_retriever(
    real_index_records: list[IndexRecord], real_embedding_provider: EmbeddingProvider
) -> Retriever:
    return Retriever(real_index_records, real_embedding_provider)


@pytest.fixture
def real_evidence_assembler(real_retriever: Retriever) -> EvidenceAssembler:
    # Function-scoped (cheap: wraps the session-scoped retriever) so each
    # test gets its own EvidenceAssemblyOptions defaults without leaking
    # state between tests.
    return EvidenceAssembler(real_retriever, options=EvidenceAssemblyOptions(top_k=8))
