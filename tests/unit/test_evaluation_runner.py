"""Regression coverage for the Phase 7 evaluation suite itself.

Runs the real `evaluation/visible-cases.json` + `evaluation/custom-cases.json`
through `run_suite` against the real evidence/order layers (no real LLM,
no network — see `evaluation/harness.py`). This pins the suite's overall
pass rate so a future regression in BUG-001/002/003/004's fixes (or in
the suite's own scripted answers/assertions) is caught here, not just by
manually re-running `python -m aster_row_agent.cli eval`.
"""

from __future__ import annotations

from pathlib import Path

from aster_row_agent.config import Settings
from aster_row_agent.evaluation.runner import run_suite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


def test_evaluation_suite_passes_completely() -> None:
    """BUG-004 (docs/architecture.md §22) was the suite's one previously
    documented, deliberately-unfixed failure (`final-sale-damaged-exception`)
    — now fixed, so the full suite is expected to pass with zero
    failures."""
    settings = Settings()
    report = run_suite(
        [EVALUATION_DIR / "visible-cases.json", EVALUATION_DIR / "custom-cases.json"],
        settings,
        run_label="pytest-regression",
    )

    failing_ids = {r.case_id for r in report.results if not r.passed}
    assert failing_ids == set()
    assert report.total == 28


def test_evaluation_suite_is_deterministic_across_runs() -> None:
    settings = Settings()
    case_files = [EVALUATION_DIR / "visible-cases.json", EVALUATION_DIR / "custom-cases.json"]

    first = run_suite(case_files, settings, run_label="run-1")
    second = run_suite(case_files, settings, run_label="run-2")

    first_outcomes = {r.case_id: r.passed for r in first.results}
    second_outcomes = {r.case_id: r.passed for r in second.results}
    assert first_outcomes == second_outcomes
