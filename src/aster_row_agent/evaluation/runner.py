"""Loads evaluation case files, runs each case against the real Agent
pipeline (via `harness.py`), and evaluates its `expect` block
(via `assertions.py`) into a structured report.

No case file's exact prose is asserted byte-for-byte anywhere in this
module or its dependents — see `assertions.py`'s module docstring.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aster_row_agent.agent.llm import DEFAULT_MODEL, LLMResponse
from aster_row_agent.config import Settings
from aster_row_agent.evaluation.assertions import AssertionResult, evaluate_expect
from aster_row_agent.evaluation.harness import (
    CaseObservation,
    RecordingEvidenceAssembler,
    TurnObservation,
    run_session,
)
from aster_row_agent.evaluation.scripts import SCRIPTS
from aster_row_agent.orders.repository import OrderRepository
from aster_row_agent.rag.embeddings import FastEmbedProvider
from aster_row_agent.rag.index import load_index_records
from aster_row_agent.rag.retrieval import Retriever
from aster_row_agent.rag.retrieval_models import EvidenceAssemblyOptions


@dataclass
class CaseResult:
    case_id: str
    category: str
    assertions: list[AssertionResult] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(a.passed for a in self.assertions)


@dataclass
class EvalReport:
    run_label: str
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return self.total - self.passed_count

    def category_breakdown(self) -> dict[str, tuple[int, int]]:
        """category -> (passed, total), insertion-ordered by first appearance."""
        breakdown: dict[str, tuple[int, int]] = {}
        for result in self.results:
            passed, total = breakdown.get(result.category, (0, 0))
            breakdown[result.category] = (passed + int(result.passed), total + 1)
        return breakdown


def _canned(answer: str, cited_filenames: tuple[str, ...]) -> LLMResponse:
    payload = json.dumps({"answer": answer, "cited_filenames": list(cited_filenames)})
    return LLMResponse(raw_text=payload, model=DEFAULT_MODEL)


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = data["cases"]
    return cases


def _case_sessions(case: dict[str, Any]) -> list[list[str]]:
    """Normalize a case into one or more "sessions", each a list of raw
    user message strings. Most cases are single-session (`messages`);
    `sessions` (a list of independent message lists) is used only for
    session-isolation cases, where the assertions care about the state
    of the *last* session after an *earlier*, independent session ran."""
    if "sessions" in case:
        return [[m["content"] for m in session] for session in case["sessions"]]
    return [[m["content"] for m in case["messages"]]]


def run_one_case(
    case: dict[str, Any],
    make_evidence_assembler: Callable[[], RecordingEvidenceAssembler],
    order_repository: OrderRepository,
) -> CaseResult:
    case_id = case["id"]
    category = case.get("category", "uncategorized")
    scripted_responses = [_canned(answer, cited) for answer, cited in SCRIPTS.get(case_id, [])]

    try:
        turns: list[TurnObservation] = []
        for index, messages in enumerate(_case_sessions(case)):
            turns = run_session(
                messages,
                evidence_assembler=make_evidence_assembler(),
                order_repository=order_repository,
                scripted_responses=scripted_responses,
                session_id=f"eval-{case_id}-{index}",
            )
        observation = CaseObservation(case_id=case_id, turns=turns)
        assertions = evaluate_expect(case.get("expect", {}), observation)
        return CaseResult(case_id=case_id, category=category, assertions=assertions)
    except Exception as exc:  # noqa: BLE001 - a case that crashes is a FAIL, not a runner crash
        return CaseResult(case_id=case_id, category=category, error=f"{type(exc).__name__}: {exc}")


def run_suite(case_files: list[Path], settings: Settings, *, run_label: str) -> EvalReport:
    provider = FastEmbedProvider(settings.embedding_model_name, settings.embedding_cache_dir)
    records = load_index_records(settings.index_dir)
    retriever = Retriever(records, provider)
    options = EvidenceAssemblyOptions(top_k=8)
    order_repository = OrderRepository.load(settings.orders_file)

    def make_evidence_assembler() -> RecordingEvidenceAssembler:
        return RecordingEvidenceAssembler(retriever, options=options)

    all_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in case_files:
        for case in load_cases(path):
            if case["id"] in seen_ids:
                raise ValueError(f"duplicate case id across case files: {case['id']!r}")
            seen_ids.add(case["id"])
            all_cases.append(case)

    results = [run_one_case(case, make_evidence_assembler, order_repository) for case in all_cases]
    return EvalReport(run_label=run_label, results=results)
